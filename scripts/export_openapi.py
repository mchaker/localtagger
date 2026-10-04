"""Write the API contract (FastAPI's OpenAPI schema) to openapi.json.

openapi.json is committed so Farterrogator can generate its client types from
a pinned copy of it. CI runs this with --check and fails when the committed
file is stale, so every API change shows up as a diff of the contract.

    python scripts/export_openapi.py           # rewrite openapi.json
    python scripts/export_openapi.py --check   # exit 1 if it is out of date
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "openapi.json"


def render() -> str:
    # create_app() prints the enabled models; keep that out of the output.
    with contextlib.redirect_stdout(io.StringIO()):
        from app.main import create_app

        schema = create_app().openapi()
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="fail if --out is stale")
    args = parser.parse_args()

    content = render()
    if args.check:
        current = args.out.read_text() if args.out.exists() else ""
        if current != content:
            print(
                f"{args.out.name} is out of date. Run `python scripts/export_openapi.py` "
                "and commit the result.",
                file=sys.stderr,
            )
            return 1
        print(f"{args.out.name} is up to date.")
        return 0

    args.out.write_text(content)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
