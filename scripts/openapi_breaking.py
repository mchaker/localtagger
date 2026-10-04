"""List changes between two openapi.json files that would break Farterrogator.

    python scripts/openapi_breaking.py OLD.json NEW.json

Breaking = a route or method removed, a parameter removed or newly required,
a request field newly required, or a response field removed or retyped.
Additions are fine. Exits 1 when anything breaking is found.
"""

import json
import sys
from typing import Dict, List


def _resolve(spec: dict, schema: dict) -> dict:
    ref = schema.get("$ref")
    if ref:
        return spec["components"]["schemas"][ref.rsplit("/", 1)[-1]]
    return schema


def _type(spec: dict, schema: dict) -> str:
    schema = _resolve(spec, schema)
    if "anyOf" in schema:
        return "|".join(sorted(_type(spec, s) for s in schema["anyOf"]))
    if schema.get("type") == "array":
        return f"array<{_type(spec, schema.get('items', {}))}>"
    return schema.get("type", "object")


def _response_fields(spec: dict, op: dict) -> Dict[str, str]:
    """Field name -> type of the 200 JSON body (array items unwrapped)."""
    content = op.get("responses", {}).get("200", {}).get("content", {})
    schema = content.get("application/json", {}).get("schema")
    if not schema:
        return {}
    schema = _resolve(spec, schema)
    if schema.get("type") == "array":
        schema = _resolve(spec, schema.get("items", {}))
    return {name: _type(spec, prop) for name, prop in schema.get("properties", {}).items()}


def _request_required(spec: dict, op: dict) -> set:
    content = op.get("requestBody", {}).get("content", {})
    required = set()
    for media in content.values():
        required |= set(_resolve(spec, media.get("schema", {})).get("required", []))
    return required


def breaking_changes(old: dict, new: dict) -> List[str]:
    problems = []
    for path, old_methods in old.get("paths", {}).items():
        new_methods = new.get("paths", {}).get(path)
        if new_methods is None:
            problems.append(f"{path}: route removed")
            continue
        for method, old_op in old_methods.items():
            new_op = new_methods.get(method)
            where = f"{method.upper()} {path}"
            if new_op is None:
                problems.append(f"{where}: method removed")
                continue

            old_params = {p["name"]: p for p in old_op.get("parameters", [])}
            new_params = {p["name"]: p for p in new_op.get("parameters", [])}
            for name in old_params.keys() - new_params.keys():
                problems.append(f"{where}: parameter '{name}' removed")
            for name, param in new_params.items():
                was_required = old_params.get(name, {}).get("required", False)
                if param.get("required") and not was_required:
                    problems.append(f"{where}: parameter '{name}' is now required")

            for name in _request_required(new, new_op) - _request_required(old, old_op):
                problems.append(f"{where}: request field '{name}' is now required")

            old_fields = _response_fields(old, old_op)
            new_fields = _response_fields(new, new_op)
            for name, kind in old_fields.items():
                if name not in new_fields:
                    problems.append(f"{where}: response field '{name}' removed")
                elif new_fields[name] != kind:
                    problems.append(
                        f"{where}: response field '{name}' changed from {kind} to {new_fields[name]}"
                    )
    return problems


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    with open(sys.argv[1]) as f:
        old = json.load(f)
    with open(sys.argv[2]) as f:
        new = json.load(f)
    problems = breaking_changes(old, new)
    for problem in problems:
        print(f"BREAKING: {problem}")
    if not problems:
        print("No breaking API changes.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
