FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

# System deps for OpenCV (used by imgutils image processing)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies.
# - dghs-imgutils powers the WD14 v3 and Camie v2 taggers (ONNX).
# - timm powers the animetimm dbv4 taggers (safetensors, PyTorch from base image).
# - numpy<2.0 avoids binary incompatibility with the PyTorch/ONNX wheels.
COPY requirements.txt .
# - PyPI's onnxruntime-gpu 1.18.0 is built for CUDA 11, so replace it with the
#   CUDA 12 / cuDNN 8 build of the same version from ONNX Runtime's feed.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --force-reinstall --no-deps "onnxruntime-gpu==1.18.0" \
       --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/

# Make the base image's CUDA libs discoverable by ONNX Runtime: cuBLAS, cuFFT,
# cuRAND and cudart 12 live in the conda env, cuDNN 8 is bundled with PyTorch.
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/conda/lib:/opt/conda/lib/python3.10/site-packages/torch/lib

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
