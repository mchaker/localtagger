FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps for OpenCV (used by imgutils image processing)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies.
# - dghs-imgutils powers the WD14 v3 and Camie v2 taggers (ONNX).
# - timm powers the animetimm dbv4 taggers (safetensors, PyTorch from base image).
# - numpy<2.0 avoids binary incompatibility with the PyTorch/ONNX wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "onnxruntime-gpu==1.17.1" \
       --extra-index-url https://aiinfra.pkgs.visualstudio.com/Public/packages/onnxruntime-cuda-12

# Make the base image's CUDA libs discoverable by ONNX Runtime
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib/python3.10/site-packages/nvidia/cublas/lib:/opt/conda/lib/python3.10/site-packages/nvidia/cudnn/lib

# Models download to the HF cache on first use. Mount a volume here (and set
# HF_HOME) to persist them across restarts. HF_TOKEN is required for the gated
# animetimm models.
ENV HF_HOME=/root/.cache/huggingface

COPY app/ ./app/
COPY scripts/ ./scripts/

# Optional: bake all model weights into the image so the runtime needs no token.
# Uses a BuildKit secret so the token is never written to a layer. The token's
# account must have accepted the gated animetimm terms first. Enable with:
#   DOCKER_BUILDKIT=1 docker build --secret id=hf_token,src=./hf_token.txt .
# then run the container with HF_HUB_OFFLINE=1 (no token needed).
# RUN --mount=type=secret,id=hf_token \
#     HF_TOKEN="$(cat /run/secrets/hf_token)" python scripts/prefetch_models.py

EXPOSE 8000

CMD ["python", "-m", "app.main"]
