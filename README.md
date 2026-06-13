# LAN Image Interrogator

A FastAPI microservice for tagging anime/illustration images with **Danbooru
tags** across multiple model families, designed for Kubernetes deployment with
GPU support. It is the backend for *Farterrogator*.

Inference is powered by [`dghs-imgutils`](https://dghs-imgutils.deepghs.org/)
(WD14 v3, Pixai v0.9 and Camie v2, ONNX) and
[`timm`](https://github.com/huggingface/pytorch-image-models)
(animetimm dbv4, PyTorch/safetensors), plus the **Kaloscope 2.0** artist-style
classifier (ONNX).

## Features
-   **Multi-Model Tagging** — pick a model per request via `?model=<id>`. Models
    are grouped for the picker (`group` → `label`):
    -   **WD Tagger v3**: `wd-eva02-large-v3` (EVA02), `wd-swinv2-v3` (SwinV2),
        `pixai-v0.9` (Pixai v0.9).
    -   **Animetimm dbv4**: `animetimm-mobilenetv4` (MobileNetV4),
        `animetimm-swinv2-base` (SwinV2), `animetimm-caformer-b36` (CAFormer),
        `animetimm-eva02-large` (EVA02), `animetimm-convnextv2-huge` (ConvNeXt).
    -   **Camie v2**: `camie-v2` (Camie).
-   **Config-driven model catalog** — add models by editing
    [`app/models.yaml`](app/models.yaml); no code changes needed.
-   **Model discovery** — `GET /models` lets the frontend list models without a
    backend redeploy.
-   **On-Demand Loading** — models load lazily on first use.
-   **Kaloscope artist classifier** — `POST /kaloscope/infer`.
-   **Kubernetes Ready** — health checks and GPU manifests included.

## Architecture

```
app/
  main.py            # FastAPI factory + entrypoint
  config.py          # env-driven Settings (HF cache/endpoint/token, device, enabled models)
  catalog.py         # parses models.yaml into the model registry
  models.yaml        # editable model catalog
  manager.py         # lazy model construction/loading
  formatting.py      # tag-string formatting + hallucination filter
  image_utils.py     # image loading helpers
  schemas.py         # pydantic response models
  backends/          # wd14, pixai, camie, animetimm, kaloscope inference
  routers/           # interrogate, catalog (/models), kaloscope
```

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ENABLED_MODELS` | models marked `default` in `models.yaml` | Comma-separated model ids to enable. |
| `HF_HOME` | `~/.cache/huggingface` | Where models are downloaded/cached. |
| `HF_ENDPOINT` | — | Point at a HuggingFace mirror for model hosting. |
| `HF_TOKEN` | — | **Required for the gated animetimm models** (the repos require agreeing to share your email). Not needed at runtime if you prefetch — see below. |
| `HF_HUB_OFFLINE` | — | Set to `1` to serve entirely from the cache and never contact HuggingFace (and never need a token). Requires all enabled models to be prefetched first. |
| `DEVICE` | `auto` | `cuda`, `cpu`, or `auto` (used by the animetimm PyTorch backend). |
| `CATALOG_PATH` | `app/models.yaml` | Path to the model catalog file. |

> **animetimm models are gated.** Accept the terms on each model's HuggingFace
> page first (a token alone cannot — the token's account must have clicked
> "Agree and access"), then provide an `HF_TOKEN` so the backend can download them.

### Prefetch weights with a throwaway token

To run the service without a standing HuggingFace token, download every enabled
model once into a persistent `HF_HOME`, then serve in offline mode:

```bash
# 1. one-time, with a token whose account has accepted the animetimm terms:
HF_TOKEN=hf_xxx HF_HOME=/data/hf python scripts/prefetch_models.py

# 2. revoke the token, then run offline (no token, no network needed):
HF_HUB_OFFLINE=1 HF_HOME=/data/hf python -m app.main
```

Point `HF_HOME` at the same volume the service mounts so the warmed cache is the
cache it reads. `HF_HUB_OFFLINE=1` is what lets you drop the token: without it,
`huggingface_hub` re-validates the gated repos with an authenticated request on
every load even when the weights are already cached.

## Kubernetes Deployment

### 1. Build & Push Image
```bash
docker build -t ghcr.io/mchaker/localtagger:main .
docker push ghcr.io/mchaker/localtagger:main
```

### 2. Deploy
```bash
# (optional) token for gated animetimm models
kubectl create secret generic hf-token --from-literal=token=hf_xxx
kubectl apply -f k8s/
```

### Deployment Notes
-   **GPU**: the deployment requests `nvidia.com/gpu: 1` — ensure the NVIDIA
    Container Toolkit is installed on your nodes.
-   **Model Caching**: models download to `HF_HOME` on first use (several GB).
    Mount a PVC there to persist across restarts — see the commented
    `volumeMounts`/`volumes`/`HF_TOKEN` blocks in `k8s/deployment.yaml`.
-   **Health Check**: `GET /health` for liveness/readiness probes.

---

## API Usage

### `GET /models` — list available models

```json
{
  "models": [
    {
      "id": "wd-swinv2-v3",
      "label": "SwinV2",
      "description": "fast and balanced wd",
      "group": "WD Tagger v3",
      "family": "wd14",
      "recommended": true,
      "gated": false,
      "loaded": false,
      "default_threshold": 0.35,
      "default_character_threshold": 0.85
    }
  ]
}
```

Build a two-level picker from each model's `group` (heading) and `label` (item),
with `description` as the subtitle. The frontend should drive its picker from
this endpoint. Strings are
stable English keys/labels; **i18n is owned by the frontend** (translate by `id`).

### `POST /interrogate` — tag uploaded images

Supports single image and batch processing.

#### Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `file` | File(s) | Required | Image file(s). Send multiple (same field name) for batching. |
| `model` | string | first enabled model | Model id from `/models`. |
| `threshold` | float | `0.35` | General-tag confidence threshold. |
| `character_threshold` | float | `0.85` | Character-tag confidence threshold. |
| `output_format` | string | `"json"` | `"zip"` for a dataset download, `"json"` for an API response. |
| `trigger_word` | string | `""` | Optional word to prepend to tags (e.g. `sks_person`). |
| `random_order` | boolean | `false` | Shuffle tags (useful for LoRA training). |
| `use_spaces` | boolean | `false` | Use spaces instead of underscores in tags. |
| `use_escape` | boolean | `true` | Escape special characters (parentheses). |
| `include_ranks` | boolean | `false` | Append `(tag:score)` ranks to the tag string. |

The legacy routes `/interrogate/{eva,pixai,camie,taggerine}` still work and map
to concrete models for backward compatibility.

### `GET /interrogate` — tag images by URL

`?url=<img1>&url=<img2>&model=<id>&threshold=0.35`

### `POST /kaloscope/infer` — artist-style classification

`file` (image) + `top_k` (default 10) → `{ "artists": [{"name", "score"}], "model": "kaloscope-2.0" }`

---

## Frontend Integration Guide

### Response Structure (JSON)

`POST/GET /interrogate` with `output_format="json"` returns an **array of objects**:

```json
[
  {
    "tags": { "1girl": 0.99, "solo": 0.97, "smile": 0.82 },
    "tag_string": "1girl, solo, smile",
    "rating": { "general": 0.9 },
    "character": {},
    "model": "wd-swinv2-v3"
  }
]
```

*   **`tags`**: general + character tags merged, as `{name: score}` (0.0–1.0).
*   **`tag_string`**: comma-separated, ready for display or `.txt` files.
*   **`rating` / `character`**: richer breakdown for newer clients (safe to ignore).

### Batch Processing

```javascript
const formData = new FormData();
files.forEach((file) => formData.append("file", file)); // same field name

const params = new URLSearchParams({ model: "wd-swinv2-v3", threshold: "0.35" });
await fetch(`http://localhost:8000/interrogate?${params}`, { method: "POST", body: formData });
```

### Downloading Datasets (ZIP)

```javascript
const params = new URLSearchParams({
  model: "wd-eva02-large-v3",
  output_format: "zip",
  trigger_word: "my_trigger",
  random_order: "true",
});
const response = await fetch(`http://localhost:8000/interrogate?${params}`, {
  method: "POST",
  body: formData,
});
if (response.ok) {
  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "dataset.zip";
  a.click();
  window.URL.revokeObjectURL(url);
}
```

---

## Local Development

```bash
pip install -r requirements.txt
# optional: export HF_TOKEN=hf_xxx   # for gated animetimm models
python -m app.main                   # serves on :8000

# smoke tests
python test_api.py --image some.jpg --model wd-swinv2-v3
python test_batch.py --model wd-swinv2-v3 img1.jpg img2.jpg
```

Or with Docker:

```bash
./build_and_run.sh
# Or manually:
docker build -t lan-interrogator .
docker run --gpus all -d -p 8000:8000 -e HF_TOKEN=hf_xxx --name interrogator lan-interrogator
```
