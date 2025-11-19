# LAN Image Interrogator

This project hosts EVA-CLIP and WD14 (PixAI) image interrogation models as a unified FastAPI microservice running in a Docker container.

## Prerequisites

- Docker
- NVIDIA GPU with drivers installed
- NVIDIA Container Toolkit (for `--gpus all` support)

## Setup

1.  **Build and Run:**
    You can use the provided script to build and run the container:
    ```bash
    ./build_and_run.sh
    ```

    Or run the commands manually:
    ```bash
    # Build
    sudo docker build -t lan-interrogator .

    # Run
    sudo docker run --gpus all -d -p 8000:8000 --name interrogator --restart always lan-interrogator
    ```

## Usage

### API Endpoints

The service supports **POST** (file upload) and **GET** (image URL) for all models.

#### 1. VIT (ViT-L-14/openai)
Standard CLIP model.
-   **URL:** `/interrogate/vit`
-   **Parameters:**
    -   `mode`: "fast" (caption only) or "best" (caption + tags). Default: "fast".
    -   `url` (GET only): URL of the image.
    -   `file` (POST only): Image file.

#### 2. EVA (ViT-g-14/laion2b_s12b_b42k)
Large, high-accuracy CLIP model. **Warning:** Swapping to/from this model takes time (~30s) and consumes significant VRAM.
-   **URL:** `/interrogate/eva`
-   **Parameters:** Same as VIT.

#### 3. PixAI (WD14)
Anime/Illustration tagger.
-   **URL:** `/interrogate/pixai`
-   **Parameters:**
    -   `threshold`: Confidence threshold (0.0-1.0). Default: 0.35.
    -   `url` (GET only): URL of the image.
    -   `file` (POST only): Image file.

### Examples

**Upload an image (VIT):**
```bash
curl -X POST -F "file=@test.jpg" "http://localhost:8000/interrogate/vit?mode=best"
```

**Interrogate via URL (EVA):**
```bash
curl "http://localhost:8000/interrogate/eva?url=https://example.com/image.png"
```

**Get Anime Tags (PixAI):**
```bash
curl -X POST -F "file=@waifu.png" "http://localhost:8000/interrogate/pixai?threshold=0.5"
```

## Kubernetes Deployment

This service is ready for Kubernetes deployment with GPU support.

1.  **Build and Push Image:**
    You need to build the Docker image and push it to a registry accessible by your cluster (e.g., GHCR, Docker Hub).
    ```bash
    docker build -t ghcr.io/mooshieblob1/localtagger:latest .
    docker push ghcr.io/mooshieblob1/localtagger:latest
    ```

2.  **Update Manifest:**
    Edit `k8s/deployment.yaml` and replace the `image` field with your image URL.

3.  **Deploy:**
    ```bash
    kubectl apply -f k8s/
    ```

### DevOps Notes (For Astro)

1.  **Image Registry**: The `k8s/deployment.yaml` uses a placeholder image (`ghcr.io/mooshieblob1/localtagger:latest`). Please build the image and update the manifest to point to your internal registry if needed.
2.  **GPU Requirements**: The deployment requests `nvidia.com/gpu: 1`. Ensure nodes have NVIDIA drivers and the **NVIDIA Container Toolkit** installed.
3.  **Model Caching (Optimization)**: Models are downloaded to `/root/.cache/huggingface` on first use. To avoid re-downloading (5GB+) on pod restarts, consider mounting a **Persistent Volume (PVC)** to this path.
