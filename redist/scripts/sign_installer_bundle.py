#!/usr/bin/env python3
"""Attach an OpenSSL Ed25519 signature to an installer-bundle descriptor."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Sequence

from installer_bundle_contract import (
    canonical_json_bytes,
    validate_installer_bundle_descriptor,
)
from package_installer_bundle import atomic_bytes
from sign_pack_manifest import SigningError, sign_with_openssl


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("descriptor", type=Path)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--openssl", default="openssl")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def sign_descriptor(
    descriptor_path: Path,
    private_key: Path,
    key_id: str,
    openssl: str = "openssl",
    output: Path | None = None,
) -> Path:
    descriptor_path = descriptor_path.resolve(strict=True)
    if not descriptor_path.name.endswith(".descriptor.json"):
        raise ValueError("descriptor name must end with .descriptor.json")
    document = json.loads(descriptor_path.read_text(encoding="utf-8"))
    validate_installer_bundle_descriptor(document, require_signature=False)
    if document["signatures"]:
        raise ValueError("descriptor is already signed; regenerate it first")
    payload = canonical_json_bytes(document["signed"])
    signature_input = descriptor_path.with_name(
        descriptor_path.name.removesuffix(".descriptor.json") + ".signed.json"
    )
    if signature_input.read_bytes() != payload:
        raise ValueError("canonical signature input does not match the descriptor")
    signature = sign_with_openssl(
        payload, private_key.resolve(strict=True), openssl
    )
    document["signatures"] = [
        {
            "key_id": key_id,
            "algorithm": "ed25519",
            "value": base64.urlsafe_b64encode(signature)
            .decode("ascii")
            .rstrip("="),
        }
    ]
    validate_installer_bundle_descriptor(document)
    destination = (output or descriptor_path).resolve()
    atomic_bytes(
        destination, (json.dumps(document, indent=2) + "\n").encode("utf-8")
    )
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        output = sign_descriptor(
            args.descriptor,
            args.private_key,
            args.key_id,
            args.openssl,
            args.output,
        )
    except (OSError, ValueError, SigningError, json.JSONDecodeError) as error:
        print(f"[ERROR] {error}")
        return 1
    print(f"[OK] Signed installer bundle: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
