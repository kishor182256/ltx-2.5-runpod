FROM nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    LTX_REPO_DIR=/opt/LTX-2 \
    LTX_MODEL_DIR=/workspace/models/ltx-2.5 \
    LTX_OUTPUT_DIR=/workspace/outputs \
    HF_HOME=/workspace/hf-cache \
    PATH=/opt/LTX-2/.venv/bin:/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git ffmpeg libgl1 libglib2.0-0 python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Pin the upstream LTX source revision used to build this service image.
ARG LTX_REPO_REF=e98f6d927d83c92ef2be6aec76c8440b19306e39
RUN git clone https://github.com/Lightricks/LTX-2.git /opt/LTX-2 \
    && git -C /opt/LTX-2 checkout --detach "${LTX_REPO_REF}" \
    && cd /opt/LTX-2 \
    && uv sync --frozen --package ltx-pipelines --no-dev \
    && uv pip install --python /opt/LTX-2/.venv/bin/python fastapi uvicorn huggingface_hub

WORKDIR /app
COPY service/app.py /app/app.py
COPY service/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
