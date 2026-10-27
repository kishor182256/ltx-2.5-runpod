from __future__ import annotations

import hmac
import os
import secrets
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator


MODEL_DIR = Path(os.environ.get("LTX_MODEL_DIR", "/workspace/models/ltx-2.5"))
OUTPUT_DIR = Path(os.environ.get("LTX_OUTPUT_DIR", "/workspace/outputs"))
LTX_ROOT = Path(os.environ.get("LTX_REPO_DIR", "/opt/LTX-2"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "transformer": MODEL_DIR / "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
    "text_encoder": MODEL_DIR / "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
    "video_vae": MODEL_DIR / "vae/ltx-2.5-video-vae-bf16.safetensors",
    "audio_vae": MODEL_DIR / "vae/ltx-2.5-audio-vae-bf16.safetensors",
    "duration_head": MODEL_DIR / "model_patches/ltx-2.5-duration-head-bf16.safetensors",
    "spatial_upsampler": MODEL_DIR / "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
}

app = FastAPI(title="LTX-2.5 RunPod Service", version="1.0.0")
bearer = HTTPBearer(auto_error=False)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ltx-render")


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2000)
    width: int = Field(default=768, ge=512, le=1280)
    height: int = Field(default=512, ge=512, le=1280)
    num_frames: int = Field(default=49, ge=9, le=249)
    frame_rate: float = Field(default=24, ge=8, le=60)
    seed: int = Field(default=-1, ge=-1, le=2147483647)

    @field_validator("prompt")
    @classmethod
    def nonempty_prompt(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Prompt cannot be empty")
        return value

    @field_validator("width", "height")
    @classmethod
    def multiples_of_32(cls, value: int) -> int:
        if value % 32:
            raise ValueError("Width and height must be divisible by 32")
        return value

    @field_validator("num_frames")
    @classmethod
    def causal_frame_count(cls, value: int) -> int:
        if (value - 1) % 8:
            raise ValueError("Frame count must be 8 × k + 1 (for example 49 or 97)")
        return value


def require_token(credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]) -> None:
    expected = os.environ.get("LTX_SERVICE_TOKEN", "")
    if len(expected) < 16:
        raise HTTPException(status_code=503, detail="LTX_SERVICE_TOKEN is not configured")
    supplied = credentials.credentials if credentials else ""
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid service token")


def get_job(job_id: str) -> dict:
    with jobs_lock:
        job = jobs.get(job_id)
        return dict(job) if job else {}


def update_job(job_id: str, **updates) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(updates)


def cli_command(job_id: str, request: GenerateRequest) -> list[str]:
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    command = [
        "python", "-m", "ltx_pipelines.distilled",
        "--transformer-path", str(FILES["transformer"]),
        "--text-encoder-path", str(FILES["text_encoder"]),
        "--video-vae-path", str(FILES["video_vae"]),
        "--audio-vae-path", str(FILES["audio_vae"]),
        "--duration-head-path", str(FILES["duration_head"]),
        "--spatial-upsampler-path", str(FILES["spatial_upsampler"]),
        "--prompt", request.prompt,
        "--width", str(request.width),
        "--height", str(request.height),
        "--num-frames", str(request.num_frames),
        "--frame-rate", str(request.frame_rate),
        "--seed", str(request.seed if request.seed >= 0 else secrets.randbelow(2147483647)),
        "--output-path", str(output_path),
    ]
    offload = os.environ.get("LTX_OFFLOAD", "none").strip().lower()
    if offload in ("cpu", "disk"):
        command.extend(["--offload", offload])
    quantization = os.environ.get("LTX_QUANTIZATION", "").strip()
    if quantization:
        command.extend(["--quantization", quantization])
    return command


def run_generation(job_id: str, request: GenerateRequest) -> None:
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    log_path = OUTPUT_DIR / f"{job_id}.log"
    update_job(job_id, status="running", message="Rendering on the RunPod GPU…")
    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            completed = subprocess.run(
                cli_command(job_id, request),
                cwd=LTX_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            detail = log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
            update_job(job_id, status="error", message="LTX generation failed. " + detail.strip())
            output_path.unlink(missing_ok=True)
            return
        if not output_path.is_file() or output_path.stat().st_size == 0:
            update_job(job_id, status="error", message="LTX finished without writing a video file.")
            return
        update_job(
            job_id,
            status="done",
            message="Generation complete. Download the video.",
            filename=output_path.name,
            file_size=output_path.stat().st_size,
        )
    except Exception as exc:
        update_job(job_id, status="error", message=f"Could not start LTX generation: {exc}")
        output_path.unlink(missing_ok=True)
    finally:
        log_path.unlink(missing_ok=True)


@app.get("/health")
def health(_: Annotated[None, Depends(require_token)]):
    missing = [str(path) for path in FILES.values() if not path.is_file()]
    if missing:
        raise HTTPException(status_code=503, detail="LTX model files are missing on the Pod: " + ", ".join(missing))
    return {"status": "ready", "model": "LTX-2.5", "pipeline": "distilled"}


@app.post("/generate", status_code=202)
def generate(request: GenerateRequest, _: Annotated[None, Depends(require_token)]):
    job_id = uuid.uuid4().hex
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "message": "Waiting for the RunPod GPU…", "filename": None}
    worker.submit(run_generation, job_id, request)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str, _: Annotated[None, Depends(require_token)]):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **job}


@app.get("/jobs/{job_id}/video")
def job_video(job_id: str, _: Annotated[None, Depends(require_token)]):
    job = get_job(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(status_code=404, detail="Video is not ready")
    path = OUTPUT_DIR / str(job.get("filename", ""))
    if not path.is_file() or path.parent != OUTPUT_DIR:
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(path, media_type="video/mp4", filename="ltx-2.5-video.mp4")
