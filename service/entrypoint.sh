#!/usr/bin/env bash
set -euo pipefail

model_dir="${LTX_MODEL_DIR:-/workspace/models/ltx-2.5}"
mkdir -p "$model_dir" "${LTX_OUTPUT_DIR:-/workspace/outputs}" "${HF_HOME:-/workspace/hf-cache}"

model_files=(
  "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
  "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
  "vae/ltx-2.5-video-vae-bf16.safetensors"
  "vae/ltx-2.5-audio-vae-bf16.safetensors"
  "model_patches/ltx-2.5-duration-head-bf16.safetensors"
  "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
)

need_download=0
for file in "${model_files[@]}"; do
  if [[ ! -s "$model_dir/$file" ]]; then
    need_download=1
    break
  fi
done

if [[ "$need_download" == "1" ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is required. Accept the LTX-2.5 model terms on Hugging Face and set the token in the RunPod Pod template." >&2
    exit 1
  fi
  echo "Downloading the LTX-2.5 model files to the mounted RunPod volume at $model_dir ..."
  hf download Lightricks/LTX-2.5 "${model_files[@]}" --local-dir "$model_dir" --token "$HF_TOKEN"
fi

if [[ -z "${LTX_SERVICE_TOKEN:-}" || ${#LTX_SERVICE_TOKEN} -lt 16 ]]; then
  echo "Set LTX_SERVICE_TOKEN to a random secret of at least 16 characters in the RunPod Pod template." >&2
  exit 1
fi

echo "LTX-2.5 files are ready on RunPod storage. Starting the API on port 8000."
exec python -m uvicorn app:app --host 0.0.0.0 --port 8000
