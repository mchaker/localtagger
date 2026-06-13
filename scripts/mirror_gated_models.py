"""Mirror the gated models from the catalog into your own (ungated) HF repos.

The animetimm dbv4 models are gated, but GPL-3.0 licensed — so redistribution is
permitted. This downloads each gated repo once (needs a token whose account has
accepted the terms) and re-uploads it under a namespace you control with the
gate off. Afterwards, point models.yaml `repo:` at the mirrors and the service
needs no token or gate acceptance ever again.

Usage
-----
    pip install huggingface_hub
    hf auth login                      # or: export HF_TOKEN=hf_xxx  (write scope)
    python scripts/mirror_gated_models.py --target <your-username-or-org>

    # then rewrite the catalog to use your mirrors:
    python scripts/mirror_gated_models.py --target <your-username-or-org> --rewrite-catalog

Options
-------
    --target NS        Namespace (user or org) to create the mirror repos under. Required.
    --private          Create the mirror repos private instead of public.
    --rewrite-catalog  After mirroring, rewrite app/models.yaml `repo:` fields in
                       place to point at the new mirrors (and flip gated->false).
    --only ID[,ID...]  Mirror only these catalog ids (default: every gated model).
"""

import argparse
import sys

from app.catalog import load_catalog
from app.config import settings

# Keep mirrors lean: weights + the metadata our backends read, minus training logs.
IGNORE = ["*.bin", "runs/*", "**/events.out.tfevents.*", "*.tfevents.*"]


def _target_repo(target: str, source_repo: str) -> str:
    return f"{target}/{source_repo.split('/')[-1]}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Mirror gated catalog models to your own HF repos.")
    ap.add_argument("--target", required=True, help="HF namespace (user/org) for the mirrors")
    ap.add_argument("--private", action="store_true", help="create mirror repos as private")
    ap.add_argument("--rewrite-catalog", action="store_true", help="repoint models.yaml at mirrors")
    ap.add_argument("--only", default="", help="comma-separated catalog ids to mirror")
    args = ap.parse_args()

    settings.apply_hf_env()
    catalog = load_catalog(settings.catalog_path, settings.enabled_models)

    wanted = {s.strip() for s in args.only.split(",") if s.strip()}
    gated = [s for s in catalog.specs if s.gated and (not wanted or s.id in wanted)]
    if not gated:
        print("No gated models to mirror.", file=sys.stderr)
        return 1

    from huggingface_hub import HfApi, snapshot_download, whoami

    try:
        me = whoami()
        print(f"Authenticated as: {me.get('name', me)}")
    except Exception:
        print("ERROR: not logged in. Run `hf auth login` or set HF_TOKEN (write scope).", file=sys.stderr)
        return 1

    api = HfApi()
    mapping = {}
    for spec in gated:
        dst = _target_repo(args.target, spec.repo)
        print(f"\n=== {spec.id}: {spec.repo} -> {dst} ===")
        local = snapshot_download(spec.repo, ignore_patterns=IGNORE, token=settings.hf_token)
        api.create_repo(dst, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=dst,
            folder_path=local,
            repo_type="model",
            commit_message=f"Mirror of {spec.repo} (GPL-3.0)",
            ignore_patterns=IGNORE,
        )
        mapping[spec.id] = dst
        print(f"  mirrored -> https://huggingface.co/{dst}")

    print("\nMirror complete. Catalog mapping:")
    for mid, dst in mapping.items():
        print(f"  {mid}: {dst}")

    if args.rewrite_catalog:
        _rewrite_catalog(mapping)
        print(f"\nRewrote {settings.catalog_path} to use mirrors (gated -> false).")
    else:
        print("\nUpdate app/models.yaml `repo:` fields to the mirrors above, or re-run "
              "with --rewrite-catalog to do it automatically.")
    return 0


def _rewrite_catalog(mapping: dict) -> None:
    """Repoint each mirrored model's `repo:` at its mirror and clear `gated`.

    Edits the YAML textually so comments/formatting/order are preserved.
    """
    import yaml

    path = settings.catalog_path
    raw = yaml.safe_load(path.read_text())
    repo_by_id = {m["id"]: m for m in raw.get("models", [])}
    lines = path.read_text().splitlines()
    current_id = None
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("- id:"):
            current_id = stripped.split("- id:", 1)[1].strip()
        if current_id in mapping:
            indent = line[: len(line) - len(line.lstrip())]
            if stripped.startswith("repo:"):
                line = f"{indent}repo: {mapping[current_id]}"
            elif stripped.startswith("gated:"):
                line = f"{indent}gated: false"
        out.append(line)
    path.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
