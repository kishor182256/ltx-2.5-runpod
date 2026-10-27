from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "generated"
OUTPUTS.mkdir(exist_ok=True)
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
MAX_BODY_BYTES = 2 * 1024 * 1024
POLL_SECONDS = 4
JOB_TIMEOUT_SECONDS = 3 * 60 * 60
RUNPOD_API = "https://api.runpod.io/v2"
SERVICE_PORT = 8000


def json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False).encode("utf-8")


def api_url(pod_id: str) -> str:
    pod_id = pod_id.strip()
    if not re.fullmatch(r"[A-Za-z0-9-]{4,80}", pod_id):
        raise ValueError("Select a valid running RunPod Pod")
    return f"https://{pod_id}-{SERVICE_PORT}.proxy.runpod.net"


def request_json(url: str, token: str = "", body: object | None = None, timeout: int = 30):
    headers = {"Accept": "application/json"}
    data = None
    if token.strip():
        headers["Authorization"] = "Bearer " + token.strip()
    if body is not None:
        data = json_bytes(body)
        headers["Content-Type"] = "application/json"
    req = Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urlopen(req, timeout=timeout) as response:
            raw = response.read()
            content_type = response.headers.get("Content-Type", "")
            if "json" in content_type or raw[:1] in (b"{", b"["):
                return json.loads(raw.decode("utf-8"))
            return raw
    except HTTPError as exc:
        detail = exc.read(12000).decode("utf-8", errors="replace")
        raise RuntimeError(f"Service returned HTTP {exc.code}: {detail[:1800]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Couldn't reach the service: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError("The request timed out. Check that the Pod is running.") from exc


def runpod_get(path: str, api_key: str, timeout: int = 30):
    return request_json(RUNPOD_API + path, api_key, timeout=timeout)


def available_pods(api_key: str) -> list[dict]:
    response = runpod_get("/pods", api_key)
    pods = response.get("pods", []) if isinstance(response, dict) else []
    result = []
    for pod in pods:
        if not isinstance(pod, dict) or str(pod.get("status", "")).upper() != "RUNNING":
            continue
        ports = pod.get("ports", [])
        normalized_ports = []
        for port in ports if isinstance(ports, list) else []:
            if isinstance(port, str):
                normalized_ports.append(port.lower().replace(" ", ""))
            elif isinstance(port, dict):
                number = port.get("privatePort") or port.get("port")
                protocol = port.get("type") or port.get("protocol")
                if number and protocol:
                    normalized_ports.append(f"{number}/{protocol}".lower().replace(" ", ""))
        gpu = pod.get("gpu") or {}
        result.append({
            "id": pod.get("id", ""),
            "name": pod.get("name") or "RunPod Pod",
            "gpu": gpu.get("id", "GPU"),
            "gpu_count": gpu.get("count", 1),
            "ports": normalized_ports,
            "has_api_port": f"{SERVICE_PORT}/http" in normalized_ports,
        })
    return result


def service_json(base_url: str, path: str, token: str, body: object | None = None, timeout: int = 30):
    return request_json(base_url + path, token, body=body, timeout=timeout)


def download_video(job_id: str, base_url: str, token: str, remote_job_id: str) -> Path:
    destination = OUTPUTS / f"{job_id[:8]}.mp4"
    request = Request(
        f"{base_url}/jobs/{remote_job_id}/video",
        headers={"Authorization": "Bearer " + token},
    )
    try:
        with urlopen(request, timeout=300) as response, destination.open("wb") as output_file:
            shutil.copyfileobj(response, output_file)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"The render finished, but the video download failed: {exc}") from exc
    return destination


def get_job(job_id: str) -> dict | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def update_job(job_id: str, **updates) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(updates)


def watch_job(job_id: str, base_url: str, token: str, remote_job_id: str) -> None:
    deadline = time.monotonic() + JOB_TIMEOUT_SECONDS
    last_error = ""
    while time.monotonic() < deadline:
        try:
            job = service_json(base_url, "/jobs/" + remote_job_id, token, timeout=60)
            status = str(job.get("status", "running")) if isinstance(job, dict) else "running"
            if status == "done":
                path = download_video(job_id, base_url, token, remote_job_id)
                update_job(
                    job_id,
                    status="done",
                    message="Finished. The video is saved on this computer.",
                    output_url=f"/media/{job_id}",
                    file_name=path.name,
                    file_size=path.stat().st_size,
                )
                return
            if status == "error":
                update_job(job_id, status="error", message=job.get("message", "RunPod generation failed."))
                return
            update_job(job_id, status=status, message=job.get("message", "Rendering on RunPod…"))
            last_error = ""
        except Exception as exc:  # Continue through brief proxy/network hiccups.
            last_error = str(exc)
            update_job(job_id, status="running", message="Waiting for RunPod… " + last_error[:220])
        time.sleep(POLL_SECONDS)
    update_job(job_id, status="error", message="Timed out waiting for the video. " + last_error[:400])


class Handler(BaseHTTPRequestHandler):
    server_version = "LTXRunPodLocal/2.0"

    def log_message(self, fmt, *args):
        # Avoid logging prompts, API keys, or service tokens.
        print("[local-ui] " + (fmt % args))

    def send_json(self, status: int, data: object) -> None:
        body = json_bytes(data)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY_BYTES:
            raise ValueError("Request is empty or too large")
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object")
        return data

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            page = (ROOT / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(page)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(page)
            return

        match = re.fullmatch(r"/api/jobs/([A-Za-z0-9-]+)", self.path)
        if match:
            job = get_job(match.group(1))
            self.send_json(200, job) if job else self.send_json(404, {"error": "Job not found"})
            return

        match = re.fullmatch(r"/media/([A-Za-z0-9-]+)", self.path)
        if match:
            job = get_job(match.group(1))
            if not job or not job.get("output_url"):
                self.send_json(404, {"error": "Video not found"})
                return
            path = OUTPUTS / str(job.get("file_name", ""))
            if not path.is_file() or path.parent != OUTPUTS:
                self.send_json(404, {"error": "Video file not found"})
                return
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "video/mp4")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Content-Disposition", f'inline; filename="{path.name}"')
            self.end_headers()
            with path.open("rb") as stream:
                shutil.copyfileobj(stream, self.wfile)
            return

        if self.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path not in ("/api/pods", "/api/test", "/api/generate"):
            self.send_json(404, {"error": "Not found"})
            return
        try:
            data = self.read_json()
            if self.path == "/api/pods":
                api_key = str(data.get("api_key", "")).strip()
                if not api_key:
                    raise ValueError("Enter your RunPod API key to find running Pods")
                pods = available_pods(api_key)
                has_port = any(pod["has_api_port"] for pod in pods)
                if not pods:
                    message = "No running Pods found. Deploy and start your LTX service Pod, then refresh."
                elif has_port:
                    message = f"Found {len(pods)} running Pod(s); at least one exposes HTTP port {SERVICE_PORT}."
                else:
                    message = f"Found {len(pods)} running Pod(s), but none reports HTTP port {SERVICE_PORT}."
                self.send_json(200, {"pods": pods, "message": message})
                return

            pod_id = str(data.get("pod_id", ""))
            base_url = api_url(pod_id)
            token = str(data.get("service_token", "")).strip()
            if len(token) < 16:
                raise ValueError("Enter the service token configured on the RunPod Pod")

            if self.path == "/api/test":
                result = service_json(base_url, "/health", token, timeout=60)
                self.send_json(200, {
                    "ok": True,
                    "message": "Connected to the LTX service on RunPod.",
                    "service": result,
                })
                return

            prompt = str(data.get("prompt", "")).strip()
            if not prompt:
                raise ValueError("Enter a prompt first")
            request_body = {
                "prompt": prompt,
                "width": int(data.get("width", 768)),
                "height": int(data.get("height", 512)),
                "num_frames": int(data.get("num_frames", 49)),
                "frame_rate": float(data.get("frame_rate", 24)),
                "seed": int(data.get("seed", -1)),
            }
            result = service_json(base_url, "/generate", token, request_body, timeout=120)
            remote_job_id = result.get("job_id") if isinstance(result, dict) else None
            if not remote_job_id:
                raise RuntimeError("The RunPod service did not return a job ID")
            job_id = uuid.uuid4().hex
            with JOBS_LOCK:
                JOBS[job_id] = {
                    "status": "queued",
                    "message": "Submitted to the LTX service on RunPod.",
                    "output_url": None,
                    "file_name": None,
                }
            threading.Thread(
                target=watch_job,
                args=(job_id, base_url, token, str(remote_job_id)),
                daemon=True,
            ).start()
            self.send_json(200, {"job_id": job_id, "status": "queued"})
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception as exc:
            self.send_json(502, {"error": str(exc)})


def main():
    host, port = "127.0.0.1", int(os.environ.get("LTX_UI_PORT", "8765"))
    server = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}"
    print(f"LTX-2.5 local prompt UI: {url}")
    print("Only this computer can connect. Close this window to stop the UI.")
    threading.Timer(0.8, lambda: __import__("webbrowser").open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
