"""Environment-driven configuration.

All settings are sourced from environment variables so the service can be
configured per deployment (Docker/Kubernetes) without code changes. The HF_*
variables in particular let operators choose *where* models are hosted and
cached, as requested in issue #2.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class Settings:
    # Where the model catalog lives. Defaults to the bundled models.yaml.
    catalog_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "CATALOG_PATH", str(Path(__file__).parent / "models.yaml")
            )
        )
    )

    # Subset of catalog model ids to expose/enable. Empty => all catalog models.
    enabled_models: List[str] = field(
        default_factory=lambda: _split_csv(os.environ.get("ENABLED_MODELS", ""))
    )

    # Torch device for the animetimm (pytorch) backend: "cuda", "cpu" or "auto".
    device: str = field(default_factory=lambda: os.environ.get("DEVICE", "auto"))

    # HuggingFace hosting / caching knobs (also exported to the process env so
    # imgutils, timm and huggingface_hub all honor them).
    hf_home: Optional[str] = field(
        default_factory=lambda: os.environ.get("HF_HOME")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
    )
    hf_endpoint: Optional[str] = field(
        default_factory=lambda: os.environ.get("HF_ENDPOINT")
    )
    hf_token: Optional[str] = field(
        default_factory=lambda: os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )

    def apply_hf_env(self) -> None:
        """Propagate HF settings into the environment before any HF import use."""
        if self.hf_home:
            os.environ.setdefault("HF_HOME", self.hf_home)
            os.environ.setdefault("HUGGINGFACE_HUB_CACHE", self.hf_home)
        if self.hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", self.hf_endpoint)
        if self.hf_token:
            os.environ.setdefault("HF_TOKEN", self.hf_token)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", self.hf_token)

    def resolve_device(self) -> str:
        """Resolve "auto" to "cuda" when available, else "cpu"."""
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


settings = Settings()
