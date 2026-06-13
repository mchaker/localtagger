# Agent Handoff — issue #2 follow-up (model catalog + gated-model mirroring)

Context for a local agent picking up this branch. Read this top-to-bottom before
touching anything; the one remaining task (mirroring gated models) is blocked
*only* by the cloud sandbox's network, not by code.

## Where this came from

GitHub issue **mchaker/localtagger#2** ("Remake localtagger with imgutils for
more models") was implemented in a prior session, then the reviewer
(AshtakaOOf) left a comment listing shortcomings:
<https://github.com/mchaker/localtagger/issues/2#issuecomment-4697463611>

> Missing stuff: Many models are missing. You used long descriptions that make no
> sense compared to what I put in the issue so use those. Would be preferable if
> naming format was: `wd tagger v3 > eva 02 > best wd tagger`, `swinv2`, `pixai
> v0.9` … / `animetimm dbv4 > convnext` … Add pixai-tagger-v0.9 to WD as
> "Pixai v0.9, eva 02 but broader".

Three concrete shortcomings, all addressed on this branch:

1. **"Many models are missing"** — root cause was that only 4 of the catalog's
   models had `default: true`, so `GET /models` (and the frontend picker) only
   surfaced those 4. Fixed by enabling **all** models by default, and **adding
   the missing Pixai v0.9** model.
2. **"Long descriptions that make no sense"** — replaced every embellished
   description with the reviewer's exact short wording from the issue.
3. **Naming hierarchy** — added a display `group` field (separate from the
   backend `family`) so the picker is two-level: **group → label (description)**.

## What changed on this branch (already committed)

### Catalog / data model
- **`app/models.yaml`** — fully restructured. 9 models, all `default: true`.
  Each has `group` (display heading), short `label` (architecture name), and the
  issue's exact `description`. Groups: **WD Tagger v3** (EVA02, SwinV2, Pixai
  v0.9), **Animetimm dbv4** (MobileNetV4, SwinV2, CAFormer, EVA02, ConvNeXt),
  **Camie v2** (Camie).
- **`app/catalog.py`** — added `group` field to `ModelSpec` (falls back to
  `label` if absent); added `pixai` to `VALID_FAMILIES`.
- **`app/schemas.py`** — `ModelInfo` now exposes `group` to the frontend.
- **`app/routers/catalog.py`** — passes `group` through in `GET /models`.

### Pixai v0.9 (new model)
- **`app/backends/pixai.py`** — new backend using imgutils' native
  `get_pixai_tags()`. imgutils serves it from a public ONNX mirror, so **no HF
  token needed** despite the source repo being gated. Produces general+character
  only (no rating); takes per-category thresholds via a dict.
- **`app/backends/__init__.py`** — registered `pixai` → `PixaiTagger`.
- **`app/routers/interrogate.py`** — fixed the legacy `/interrogate/pixai` alias
  (was a placeholder pointing at `wd-swinv2-v3`) to target the real `pixai-v0.9`.
- **`requirements.txt`** — pinned `dghs-imgutils>=0.19.0` (first version with
  `get_pixai_tags`).

### Tooling for running gated models without a standing token
- **`scripts/prefetch_models.py`** — warms every enabled model into `HF_HOME`
  via a real inference pass, so the service can then run with `HF_HUB_OFFLINE=1`
  and no token. Gated models use `HF_TOKEN`; unloads between models to bound RAM.
- **`scripts/mirror_gated_models.py`** — **THE REMAINING TASK'S TOOL.** Mirrors
  the gated animetimm models to repos under a namespace you control (legal: they
  are GPL-3.0), then optionally rewrites `models.yaml` to point at the mirrors.
- **`Dockerfile`** — copies `scripts/`, plus a commented BuildKit-secret block to
  optionally bake the cache into the image without leaking the token in a layer.
- **`README.md`** — updated model list, `GET /models` example (now shows
  `group`), a "Prefetch weights with a throwaway token" section, and an
  `HF_HUB_OFFLINE` env row.

## THE REMAINING TASK — mirror the 5 gated models (do this locally)

The animetimm dbv4 models are gated (🔒) but **GPL-3.0**, so redistribution is
permitted. The plan is to mirror them once into an ungated namespace the project
controls, then repoint the catalog — after which the service needs no token and
no gate acceptance, ever.

**Why it wasn't done in the cloud session:** the sandbox's egress IP is
edge-throttled by HuggingFace — *every* request (gated, public, even `whoami`)
returns HTTP 429. Confirmed repeatedly. Pushing ~8–10 GB of weights through a
429'd IP can't complete. It must run from a normally-rated network.

### Run it (from a local checkout, authenticated, un-throttled network)

```bash
pip install huggingface_hub
hf auth login            # write-scoped token; account must have accepted the
                         # animetimm gates (one click each, auto-approved)
python scripts/mirror_gated_models.py --target <your-username-or-org> --rewrite-catalog
```

The 5 gated repos to mirror (catalog id → source repo):
- `animetimm-mobilenetv4`     → `animetimm/mobilenetv4_conv_aa_large.dbv4-full`
- `animetimm-swinv2-base`     → `animetimm/swinv2_base_window8_256.dbv4-full`
- `animetimm-caformer-b36`    → `animetimm/caformer_b36.dbv4-full`
- `animetimm-eva02-large`     → `animetimm/eva02_large_patch14_448.dbv4-full`
- `animetimm-convnextv2-huge` → `animetimm/convnextv2_huge.dbv4-full`

`--rewrite-catalog` edits `app/models.yaml` in place: each mirrored model's
`repo:` → `<target>/<name>` and `gated: false`. Without the flag it just prints
the mapping for you to apply manually.

### Gotchas
- A token alone can't bypass the gate — the token's **account** must have clicked
  "Agree and access" on each animetimm page first (instant, auto-approved).
- Each mirror must keep `selected_tags.csv` + `preprocess.json` + the timm
  `config.json` + weights — the animetimm backend reads those. The script copies
  everything except training logs (`runs/`, tfevents) and `*.bin` duplicates.
- If you hit transient 429s on the big uploads, add retry/backoff (not yet wired
  in — see the open item below).

### Verify after mirroring
1. `python -c "from app.catalog import load_catalog; from pathlib import Path; [print(s.id, s.repo, s.gated) for s in load_catalog(Path('app/models.yaml')).specs]"`
   — the 5 animetimm rows should show your namespace and `gated=False`.
2. Each `https://huggingface.co/<target>/<name>` loads without a gate prompt.
3. `python scripts/prefetch_models.py` (no token) completes — proves the catalog
   resolves and downloads work tokenless.

## Open items / decisions for the maintainer
- **Retry/backoff** on `mirror_gated_models.py` uploads — offered, not yet added.
- Whether mirror repos should be **public or private** (`--private` flag exists).
- Whether to **drop the gated source repos** from the catalog entirely once
  mirrors exist, vs keep both.
- The `pixai-v0.9` model is grouped under **"WD Tagger v3"** per the reviewer's
  explicit request, even though its backend family is `pixai`. Intentional.

## Validation already done in-session (no network needed)
- `python -m compileall app/ scripts/` — clean.
- Catalog parses all 9 models with correct group/label/description.
- Every model builds a backend; `pixai` family registered.
- `mirror_gated_models.py` `--rewrite-catalog` tested on a temp copy: only mapped
  rows change, YAML stays valid.

What could NOT be validated here (network-blocked): actual HF downloads/uploads,
and live inference through imgutils/timm (not installed in the sandbox).
