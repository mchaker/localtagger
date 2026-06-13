"""Pre-download every enabled model into HF_HOME so the running service needs
no HuggingFace token at request time.

Why this exists
---------------
The animetimm dbv4 models are gated: downloading them needs (a) an HF token and
(b) prior acceptance of each repo's terms on the token-owning account (click
"Agree and access" on each model page once — a token alone cannot do this).

Once the weights are cached you can run the service with the token revoked, but
only in offline mode: huggingface_hub otherwise re-validates gated repos with an
authenticated request on every load. HF_HUB_OFFLINE=1 disables that check
globally, which is why this script warms *all* enabled models (public ONNX ones
included), not just the gated ones.

Usage
-----
    # one-time, with a (revocable) token that has accepted the animetimm terms:
    HF_TOKEN=hf_xxx HF_HOME=/data/hf python scripts/prefetch_models.py

    # then run the service with the token revoked and no network dependency:
    HF_HUB_OFFLINE=1 HF_HOME=/data/hf python -m app.main

Honors the same env as the app: ENABLED_MODELS, CATALOG_PATH, HF_HOME, HF_TOKEN,
HF_ENDPOINT, DEVICE. Point HF_HOME at the volume your deployment mounts so the
warmed cache is the cache the service reads.
"""

import sys

from PIL import Image

from app.catalog import load_catalog
from app.config import settings
from app.manager import ModelManager


def main() -> int:
    settings.apply_hf_env()
    catalog = load_catalog(settings.catalog_path, settings.enabled_models)

    gated = [s for s in catalog.specs if s.gated]
    if gated and not settings.hf_token:
        ids = ", ".join(s.id for s in gated)
        print(
            f"ERROR: gated models need a token but HF_TOKEN is unset: {ids}\n"
            "Set HF_TOKEN to a key whose account has accepted each repo's terms.",
            file=sys.stderr,
        )
        return 1

    manager = ModelManager(catalog)
    # A neutral image is enough to exercise the full download+inference path, so a
    # success here guarantees the runtime path is fully cached.
    dummy = Image.new("RGB", (448, 448), (128, 128, 128))

    failures = []
    for spec in catalog.specs:
        kind = "gated" if spec.gated else "public"
        print(f"[{kind}] warming {spec.id} ({spec.repo}) ...", flush=True)
        try:
            manager.get(spec.id).tag([dummy])
            manager.unload(spec.id)  # free weights between models to bound memory
        except Exception as exc:  # keep going so one bad repo doesn't block the rest
            print(f"  FAILED: {spec.id}: {exc}", file=sys.stderr)
            failures.append(spec.id)

    if failures:
        print(f"\nDone with {len(failures)} failure(s): {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\nAll {len(catalog.specs)} models cached under {settings.hf_home or 'default HF cache'}.")
    print("Start the service with HF_HUB_OFFLINE=1 (token can now be revoked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
