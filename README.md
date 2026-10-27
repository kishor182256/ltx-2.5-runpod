# LTX-2.5 on RunPod, with a local prompt page

This project gives you a small local page for writing prompts and downloading completed videos. The LTX-2.5 inference service runs in a GPU Pod on RunPod. Push this project to GitHub; GitHub Actions builds the service image and publishes it to GitHub Container Registry (GHCR). RunPod runs that image.

**Model weights never download to this computer.** On its first start, the RunPod container downloads the gated LTX-2.5 files to `/workspace` on your RunPod persistent volume. The local UI uses only the Python standard library.

The service uses the official Lightricks `ltx-pipelines` distilled pipeline. The Dockerfile pins the upstream LTX-2 source revision `e98f6d927d83c92ef2be6aec76c8440b19306e39` and installs its dependencies into the container image.

## Requirements

- A GitHub account/repository with GitHub Actions enabled.
- A RunPod account and a GPU Pod with enough GPU and system memory. LTX lists 32 GB VRAM as a minimum and recommends an A100 80 GB or H100, 64 GB+ RAM, and 200 GB+ SSD storage for its recommended configuration. See [LTX system requirements](https://docs.ltx.io/open-source-model/getting-started/system-requirements/).
- A Hugging Face account. Accept the terms for [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) and create a read token.
- Python 3.10+ on Windows to run the local prompt page.

## 1. Publish the project to GitHub

This project is prepared for [`kishor182256/ltx-2.5-runpod`](https://github.com/kishor182256/ltx-2.5-runpod) on the `main` branch. From this project folder, push the local commit with:

```powershell
git push -u origin main
```

The workflow at `.github/workflows/publish-image.yml` builds the Docker image and publishes it as:

```text
ghcr.io/kishor182256/ltx-2.5-runpod:latest
```

Wait for the GitHub Actions workflow named **Build and publish RunPod image** to finish. The GitHub repository is public, but GHCR packages are private by default. For the simplest RunPod setup, open the package in GitHub Packages and change its visibility to **Public**. If you keep the package private, configure GHCR credentials in RunPod so the Pod can pull it.

Each later push to `main` publishes a new `latest` image. Restart/redeploy the Pod to use code updates; the attached `/workspace` volume keeps model weights and outputs.

## 2. Create the RunPod Pod

1. In RunPod, create a GPU Pod using the GHCR image above. Leave its container start command empty so the included entrypoint can run.
2. Attach a persistent volume and mount it at **`/workspace`**. Use 150–200 GB or more so the model, Hugging Face cache, and generated videos persist.
3. Set the Pod's container disk to at least 40 GB for the CUDA image and Python dependencies.
4. Add these environment variables in the Pod template:

   | Variable | Value |
   |---|---|
   | `HF_TOKEN` | Your Hugging Face read token. It must have access to the accepted LTX-2.5 model. |
   | `LTX_SERVICE_TOKEN` | A private random secret, at least 16 characters. Generate one locally with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. |

   Keep these values in the RunPod template/secrets. Don't commit them to GitHub or paste them into chat.

5. Expose **HTTP port `8000`** in the RunPod Pod settings. The service uses this API port; it does not require ComfyUI or port 8188.
6. Start the Pod. On first start, the entrypoint downloads the six model files to `/workspace/models/ltx-2.5`, then starts the authenticated API. The first startup takes time because those files are large. Later starts reuse the persistent volume.

For an 80 GB A100/H100, the service defaults to BF16 without weight offloading. To trade speed for lower GPU memory use, add `LTX_OFFLOAD=cpu` and, on an FP8-capable GPU, `LTX_QUANTIZATION=fp8-cast` to the RunPod environment. After changing template variables, restart the Pod.

## 3. Run the local prompt page

1. On this computer, double-click `start.bat` (or run `python app.py` in PowerShell).
2. Enter your RunPod API key and click **Find my Pods**.
3. Enter the same `LTX_SERVICE_TOKEN` you set on RunPod, select the Pod, and click **Test selected Pod**.
4. Write a prompt and click **Generate video on RunPod**. When it finishes, the local page previews and downloads the MP4 into this project's `generated` folder.

The local page derives the RunPod connection from the selected Pod ID. It does not show or ask for a service URL. It keeps both tokens in browser memory only; refreshing clears them. The API key is used only by the local app to list Pods. The service token is used to authenticate requests to the selected Pod. Prompts go to RunPod for generation.

## Settings and files

- The local app listens only on `127.0.0.1:8765`.
- Default generation is 768×512, 49 frames (about 2 seconds at 24 fps). Supported frame counts are `8 × k + 1` (for example 49, 97, or 121); dimensions must be multiples of 32.
- Set `LTX_UI_PORT` before launching `app.py` to change the local port.
- RunPod service files live in `service/`; the container build is in `Dockerfile`; the GitHub image publishing workflow is in `.github/workflows/publish-image.yml`.
- Model files, Hugging Face cache, and RunPod copies of generated MP4s live on the attached `/workspace` volume. Local MP4s live in `generated/`.

## Troubleshooting

- **Pod isn't listed:** confirm it is a running Pod (this picker doesn't list Serverless endpoints) and re-enter the RunPod API key.
- **`8000/http missing`:** expose HTTP port 8000 on the Pod template, then click **Find my Pods** again.
- **Connection test returns 401:** the local service token must exactly match `LTX_SERVICE_TOKEN` configured on RunPod.
- **Connection test says model files are missing:** check the RunPod startup logs for Hugging Face authentication/terms errors. Confirm the HF token is set, the account accepted the model terms, and `/workspace` is mounted with enough space.
- **RunPod cannot pull the image:** make the GHCR package public or configure GHCR credentials for the Pod.
- **CUDA out of memory:** use an 80 GB GPU, generate fewer frames/lower resolution, or configure `LTX_OFFLOAD=cpu`; FP8-capable GPUs can also use `LTX_QUANTIZATION=fp8-cast`.
- **Code changes don't appear:** wait for the GitHub Actions image build to succeed, then restart/redeploy the Pod. The model volume can stay attached.
