#!/usr/bin/env python3
"""Sign a prepared CyxWiz backend-pack manifest with OpenSSL Ed25519."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backend_pack_contract import (  # noqa: E402
    canonical_json_bytes,
    validate_pack_manifest,
)


class SigningError(RuntimeError):
    """Raised when a release manifest cannot be signed safely."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach an OpenSSL Ed25519 signature to a prepared pack manifest."
    )
    parser.add_argument("manifest", type=Path, help="Unsigned .zip.manifest.json")
    parser.add_argument("--private-key", required=True, type=Path, help="OpenSSL Ed25519 private key")
    parser.add_argument("--key-id", required=True, help="Trusted release-key identity")
    parser.add_argument("--openssl", default="openssl", help="OpenSSL executable")
    parser.add_argument("--output", type=Path, help="Output manifest; defaults to replacing the input")
    return parser.parse_args(argv)


def signature_input_path(manifest_path: Path) -> Path:
    suffix = ".manifest.json"
    if not manifest_path.name.endswith(suffix):
        raise SigningError("Manifest name must end with .manifest.json")
    return manifest_path.with_name(manifest_path.name[: -len(suffix)] + ".signed.json")


def load_signature_input(manifest_path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SigningError(f"Cannot read manifest: {error}") from error
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "kind",
        "signed",
        "signatures",
    }:
        raise SigningError("Manifest is not a schema-1 signed envelope")
    if document["signatures"] != []:
        raise SigningError("Manifest is already signed; regenerate it before signing")

    payload = canonical_json_bytes(document["signed"])
    payload_path = signature_input_path(manifest_path)
    try:
        prepared_payload = payload_path.read_bytes()
    except OSError as error:
        raise SigningError(f"Cannot read canonical signature input: {error}") from error
    if prepared_payload != payload:
        raise SigningError("Canonical .signed.json does not match the manifest signed body")
    preflight = dict(document)
    preflight["signatures"] = [
        {
            "key_id": "preflight",
            "algorithm": "ed25519",
            "value": base64.urlsafe_b64encode(bytes(64)).decode("ascii").rstrip("="),
        }
    ]
    validate_pack_manifest(preflight)
    return document, payload


def run_openssl(arguments: list[str], label: str) -> None:
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, check=False)
    except OSError as error:
        raise SigningError(f"Cannot execute OpenSSL for {label}: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown OpenSSL error"
        raise SigningError(f"OpenSSL {label} failed: {detail}")


def sign_with_openssl(payload: bytes, private_key: Path, openssl: str) -> bytes:
    if not private_key.is_file():
        raise SigningError(f"Private key not found: {private_key}")
    with tempfile.TemporaryDirectory(prefix="cyxwiz-pack-sign-") as temporary:
        root = Path(temporary)
        payload_path = root / "payload.json"
        signature_path = root / "signature.bin"
        public_key_path = root / "public.pem"
        payload_path.write_bytes(payload)
        run_openssl(
            [
                openssl,
                "pkeyutl",
                "-sign",
                "-rawin",
                "-inkey",
                str(private_key),
                "-in",
                str(payload_path),
                "-out",
                str(signature_path),
            ],
            "signing",
        )
        run_openssl(
            [openssl, "pkey", "-in", str(private_key), "-pubout", "-out", str(public_key_path)],
            "public-key derivation",
        )
        run_openssl(
            [
                openssl,
                "pkeyutl",
                "-verify",
                "-rawin",
                "-pubin",
                "-inkey",
                str(public_key_path),
                "-in",
                str(payload_path),
                "-sigfile",
                str(signature_path),
            ],
            "self-verification",
        )
        signature = signature_path.read_bytes()
    if len(signature) != 64:
        raise SigningError(f"OpenSSL returned a {len(signature)}-byte signature; expected 64")
    return signature


def attach_signature(document: dict[str, Any], signature: bytes, key_id: str) -> dict[str, Any]:
    encoded = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")
    document["signatures"] = [
        {"key_id": key_id, "algorithm": "ed25519", "value": encoded}
    ]
    validate_pack_manifest(document)
    return document


def write_manifest_atomic(document: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = args.manifest.resolve()
        document, payload = load_signature_input(manifest)
        signature = sign_with_openssl(payload, args.private_key.resolve(), args.openssl)
        signed_document = attach_signature(document, signature, args.key_id)
        output = (args.output or manifest).resolve()
        write_manifest_atomic(signed_document, output)
    except (SigningError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    print(f"[OK] Signed manifest: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
