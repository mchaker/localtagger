"""List changes between two openapi.json files that would break Farterrogator.

    python scripts/openapi_breaking.py OLD.json NEW.json

Breaking = a route or method removed, a parameter removed, retyped or newly
required, a request field newly required, or a response field removed or
retyped at any depth (nested models, list items and map values included).
Additions are fine. Exits 1 when anything breaking is found.
"""

import json
import sys
from typing import List


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


def _compare_response(old_spec, old, new_spec, new, where, problems, seen=frozenset()):
    """Recursively flag removed or retyped fields between two response schemas."""
    key = (old.get("$ref"), new.get("$ref"))
    if key != (None, None):
        if key in seen:
            return
        seen = seen | {key}
    old, new = _resolve(old_spec, old), _resolve(new_spec, new)

    old_type, new_type = _type(old_spec, old), _type(new_spec, new)
    if old_type != new_type:
        problems.append(f"{where}: changed from {old_type} to {new_type}")
        return

    for name, prop in old.get("properties", {}).items():
        new_prop = new.get("properties", {}).get(name)
        if new_prop is None:
            problems.append(f"{where}.{name}: removed")
        else:
            _compare_response(old_spec, prop, new_spec, new_prop, f"{where}.{name}", problems, seen)
    if isinstance(old.get("items"), dict) and isinstance(new.get("items"), dict):
        _compare_response(old_spec, old["items"], new_spec, new["items"], f"{where}[]", problems, seen)
    if isinstance(old.get("additionalProperties"), dict) and isinstance(new.get("additionalProperties"), dict):
        _compare_response(
            old_spec, old["additionalProperties"], new_spec, new["additionalProperties"],
            f"{where}{{}}", problems, seen,
        )


def _json_200(op: dict):
    content = op.get("responses", {}).get("200", {}).get("content", {})
    return content.get("application/json", {}).get("schema")


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
                if name in old_params:
                    old_kind = _type(old, old_params[name].get("schema", {}))
                    new_kind = _type(new, param.get("schema", {}))
                    if old_kind != new_kind:
                        problems.append(
                            f"{where}: parameter '{name}' changed from {old_kind} to {new_kind}"
                        )

            for name in _request_required(new, new_op) - _request_required(old, old_op):
                problems.append(f"{where}: request field '{name}' is now required")

            old_body, new_body = _json_200(old_op), _json_200(new_op)
            if old_body and not new_body:
                problems.append(f"{where}: JSON response removed")
            elif old_body and new_body:
                _compare_response(old, old_body, new, new_body, f"{where} response", problems)
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
