#!/usr/bin/env python3
"""Prepare a diagnostic backend-pack manifest for explicit support signing."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backend_pack_contract import (  # noqa: E402
    ContractError,
    canonical_json_bytes,
    validate_pack_manifest,
)


class PromotionError(RuntimeError):
    """Raised when a support promotion candidate is not safe to prepare."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Signed diagnostic manifest")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New unsigned supported manifest; the path must not exist",
    )
    return parser.parse_args(argv)


def signature_input_path(manifest_path: Path) -> Path:
    suffix = ".manifest.json"
    if not manifest_path.name.endswith(suffix):
        raise PromotionError("Manifest name must end with .manifest.json")
    return manifest_path.with_name(
        manifest_path.name[: -len(suffix)] + ".signed.json"
    )


def _load_document(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PromotionError("Input manifest must be a regular file")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PromotionError(f"Cannot read input manifest: {error}") from error
    validate_pack_manifest(document)
    return document


def prepare_document(manifest_path: Path) -> tuple[dict[str, Any], bytes]:
    document = _load_document(manifest_path)
    canonical_input = canonical_json_bytes(document["signed"])
    try:
        prepared_input = signature_input_path(manifest_path).read_bytes()
    except OSError as error:
        raise PromotionError(f"Cannot read canonical signature input: {error}") from error
    if prepared_input != canonical_input:
        raise PromotionError(
            "Canonical .signed.json does not match the signed manifest body"
        )

    compatibility = document["signed"]["compatibility"]
    if compatibility["support_status"] != "diagnostic":
        raise PromotionError(
            "Only a diagnostic manifest can be prepared for supported promotion"
        )
    compatibility["support_status"] = "supported"
    document["signatures"] = []
    promoted_input = canonical_json_bytes(document["signed"])

    preflight = dict(document)
    preflight["signatures"] = [
        {
            "key_id": "preflight",
            "algorithm": "ed25519",
            "value": "A" * 86,
        }
    ]
    validate_pack_manifest(preflight)
    return document, promoted_input


def _write_new(path: Path, content: bytes) -> None:
    path = path.resolve()
    if path.exists():
        raise PromotionError(f"Output path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare(manifest_path: Path, output: Path) -> tuple[Path, Path]:
    source = manifest_path.resolve()
    destination = output.resolve()
    if source == destination:
        raise PromotionError("Promotion output must not replace the input manifest")
    document, signed_input = prepare_document(source)
    signature_input = signature_input_path(destination)
    if destination.exists() or signature_input.exists():
        raise PromotionError("Promotion outputs must not already exist")
    manifest_bytes = (json.dumps(document, indent=2) + "\n").encode("utf-8")
    _write_new(signature_input, signed_input)
    try:
        _write_new(destination, manifest_bytes)
    except Exception:
        signature_input.unlink(missing_ok=True)
        raise
    return destination, signature_input


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest, signature_input = prepare(args.manifest, args.output)
    except (OSError, ValueError, ContractError, PromotionError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    print(f"[OK] Supported candidate: {manifest}")
    print(f"[OK] Signature input: {signature_input}")
    print("[SIGNING REQUIRED] Promotion is not authoritative until re-signed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
