#!/usr/bin/env python3
"""Assemble a trusted CyxWiz backend-pack repository for HTTPS or offline use."""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backend_pack_contract import (  # noqa: E402
    canonical_json_bytes,
    validate_catalog,
    validate_pack_manifest,
    validate_runtime_composition,
    validate_trust_root,
)
from sign_pack_manifest import sign_with_openssl  # noqa: E402


MAX_METADATA_BYTES = 16 * 1024 * 1024
ED25519_PUBLIC_KEY_DER_PREFIX = bytes.fromhex("302a300506032b6570032100")
GITHUB_SEGMENT = re.compile(r"[A-Za-z0-9_.-]+")
GITHUB_RELEASE_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class RepositoryError(RuntimeError):
    """Raised when release inputs cannot form one trusted repository."""


@dataclass(frozen=True)
class TrustedKey:
    key_id: str
    public_key: bytes
    roles: frozenset[str]
    revoked: bool


@dataclass(frozen=True)
class PackInput:
    source_manifest: Path
    source_archive: Path
    manifest_bytes: bytes
    document: Mapping[str, Any]
    signing_key_id: str

    @property
    def signed(self) -> Mapping[str, Any]:
        return self.document["signed"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate signed pack inputs and atomically assemble the trust, "
            "catalog, manifest, and archive tree used by CyxWiz Installer."
        )
    )
    parser.add_argument(
        "--manifest",
        action="append",
        required=True,
        type=Path,
        help="Signed pack manifest; its signed archive must be beside it",
    )
    parser.add_argument("--trust-root", required=True, type=Path)
    parser.add_argument("--catalog-private-key", required=True, type=Path)
    parser.add_argument("--catalog-key-id", required=True)
    parser.add_argument(
        "--pack-key-id",
        help=(
            "Trusted pack key authorized by catalog entries; required when a "
            "manifest has more than one valid trusted pack signature"
        ),
    )
    parser.add_argument("--catalog-id", required=True)
    parser.add_argument("--generated-utc", required=True)
    parser.add_argument("--expires-utc", required=True)
    parser.add_argument("--minimum-client-version", required=True)
    parser.add_argument(
        "--base-url",
        required=True,
        help=(
            "Direct HTTPS repository root or canonical immutable GitHub "
            "Release asset root"
        ),
    )
    parser.add_argument(
        "--hosted-layout",
        choices=("nested", "flat"),
        default="nested",
        help=(
            "Hosted asset layout: nested for a conventional HTTPS tree or "
            "flat for a single release-asset directory"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--openssl", default="openssl")
    return parser.parse_args(argv)


def _read_bounded(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise RepositoryError(f"{label} is not a regular file: {path}")
    try:
        size = path.stat().st_size
    except OSError as error:
        raise RepositoryError(f"Cannot inspect {label}: {error}") from error
    if size <= 0 or size > MAX_METADATA_BYTES:
        raise RepositoryError(
            f"{label} must contain 1-{MAX_METADATA_BYTES} bytes: {path}"
        )
    try:
        return path.read_bytes()
    except OSError as error:
        raise RepositoryError(f"Cannot read {label}: {error}") from error


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_bounded(path, label)
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RepositoryError(f"Cannot parse {label}: {error}") from error
    if not isinstance(document, dict):
        raise RepositoryError(f"{label} must contain a JSON object")
    return document, raw


def _decode_base64url(value: str, expected_size: int, label: str) -> bytes:
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (ValueError, binascii.Error) as error:
        raise RepositoryError(f"{label} is not base64url: {error}") from error
    canonical = base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=")
    if len(decoded) != expected_size or canonical != value:
        raise RepositoryError(
            f"{label} must be canonical unpadded base64url with "
            f"{expected_size} decoded bytes"
        )
    return decoded


def _trusted_keys(document: Mapping[str, Any]) -> dict[str, TrustedKey]:
    validate_trust_root(document)
    result: dict[str, TrustedKey] = {}
    for entry in document["keys"]:
        key = TrustedKey(
            key_id=entry["key_id"],
            public_key=_decode_base64url(
                entry["public_key"], 32, f"trust key {entry['key_id']}"
            ),
            roles=frozenset(entry["roles"]),
            revoked=entry["revoked"],
        )
        result[key.key_id] = key
    return result


def _verify_ed25519(
    payload: bytes,
    signature: bytes,
    public_key: bytes,
    openssl: str,
) -> bool:
    with tempfile.TemporaryDirectory(prefix="cyxwiz-repository-verify-") as temporary:
        root = Path(temporary)
        payload_path = root / "payload.json"
        signature_path = root / "signature.bin"
        public_key_path = root / "public-key.der"
        payload_path.write_bytes(payload)
        signature_path.write_bytes(signature)
        public_key_path.write_bytes(ED25519_PUBLIC_KEY_DER_PREFIX + public_key)
        try:
            result = subprocess.run(
                [
                    openssl,
                    "pkeyutl",
                    "-verify",
                    "-rawin",
                    "-pubin",
                    "-keyform",
                    "DER",
                    "-inkey",
                    str(public_key_path),
                    "-in",
                    str(payload_path),
                    "-sigfile",
                    str(signature_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:
            raise RepositoryError(
                f"Cannot execute OpenSSL for signature verification: {error}"
            ) from error
    return result.returncode == 0


def _verified_envelope_key(
    document: Mapping[str, Any],
    trusted: Mapping[str, TrustedKey],
    role: str,
    openssl: str,
    preferred_key_id: str | None = None,
) -> str:
    payload = canonical_json_bytes(document["signed"])
    valid: list[str] = []
    for entry in document["signatures"]:
        key_id = entry["key_id"]
        if preferred_key_id is not None and key_id != preferred_key_id:
            continue
        key = trusted.get(key_id)
        if key is None or key.revoked or role not in key.roles:
            continue
        signature = _decode_base64url(
            entry["value"], 64, f"signature from {key_id}"
        )
        if _verify_ed25519(payload, signature, key.public_key, openssl):
            valid.append(key_id)
    if not valid:
        requested = f" {preferred_key_id}" if preferred_key_id else ""
        raise RepositoryError(
            f"Metadata has no valid trusted{requested} signature with role {role}"
        )
    if preferred_key_id is None and len(valid) != 1:
        raise RepositoryError(
            "Metadata has multiple valid pack signatures; select --pack-key-id"
        )
    return valid[0]


def verify_trusted_metadata_signature(
    document: Mapping[str, Any],
    trust_root: Path,
    role: str,
    openssl: str = "openssl",
    key_id: str | None = None,
) -> str:
    if role not in {"catalog", "pack", "installer"}:
        raise RepositoryError(f"Unsupported metadata trust role: {role}")
    trust_document, _ = _load_json(trust_root.resolve(), "trust root")
    return _verified_envelope_key(
        document,
        _trusted_keys(trust_document),
        role,
        openssl,
        key_id,
    )


def _sha256_file(path: Path) -> tuple[int, str]:
    if path.is_symlink() or not path.is_file():
        raise RepositoryError(f"Pack archive is not a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
    except OSError as error:
        raise RepositoryError(f"Cannot read pack archive {path}: {error}") from error
    return size, digest.hexdigest()


def _load_pack(
    manifest_path: Path,
    trusted: Mapping[str, TrustedKey],
    pack_key_id: str | None,
    openssl: str,
) -> PackInput:
    manifest_path = manifest_path.resolve()
    document, raw = _load_json(manifest_path, "pack manifest")
    validate_pack_manifest(document)
    signing_key_id = _verified_envelope_key(
        document, trusted, "pack", openssl, pack_key_id
    )
    archive_contract = document["signed"]["archive"]
    archive_path = manifest_path.parent / archive_contract["file_name"]
    size, digest = _sha256_file(archive_path)
    if size != archive_contract["size"]:
        raise RepositoryError(
            f"Archive size differs from signed manifest: {archive_path}"
        )
    if digest != archive_contract["sha256"]:
        raise RepositoryError(
            f"Archive SHA-256 differs from signed manifest: {archive_path}"
        )
    return PackInput(
        source_manifest=manifest_path,
        source_archive=archive_path,
        manifest_bytes=raw,
        document=document,
        signing_key_id=signing_key_id,
    )


def _validate_repository_packs(packs: Sequence[PackInput]) -> None:
    if not packs:
        raise RepositoryError("At least one signed pack manifest is required")
    by_id: dict[str, PackInput] = {}
    archive_names: set[str] = set()
    for pack in packs:
        pack_id = pack.signed["pack_id"]
        if pack_id in by_id:
            raise RepositoryError(f"Duplicate pack ID: {pack_id}")
        archive_name = pack.signed["archive"]["file_name"].casefold()
        if archive_name in archive_names:
            raise RepositoryError(
                f"Duplicate case-insensitive archive name: "
                f"{pack.signed['archive']['file_name']}"
            )
        by_id[pack_id] = pack
        archive_names.add(archive_name)

    for pack in packs:
        if pack.signed["pack_kind"] == "base":
            continue
        companion_id = pack.signed["companion_base_id"]
        base = by_id.get(companion_id)
        if base is None:
            raise RepositoryError(
                f"Pack {pack.signed['pack_id']} requires absent base {companion_id}"
            )
        validate_runtime_composition(base.document, pack.document)


def _validate_hosted_asset_names(
    packs: Sequence[PackInput], catalog_id: str, hosted_layout: str
) -> None:
    prefix = "catalogs/manifests/" if hosted_layout == "nested" else ""
    catalog_name = (
        "catalogs/current.json"
        if hosted_layout == "nested"
        else f"cyxwiz-backend-catalog-{catalog_id}.json"
    )
    observed = {catalog_name.casefold(): catalog_name}
    for pack in packs:
        for name in (
            f"{prefix}{pack.signed['pack_id']}.json",
            f"{prefix}{pack.signed['archive']['file_name']}",
        ):
            folded = name.casefold()
            previous = observed.get(folded)
            if previous is not None:
                raise RepositoryError(
                    f"Hosted release assets collide: {previous} and {name}"
                )
            observed[folded] = name


def validated_https_base_url(value: str) -> str:
    if (
        len(value) > 4000
        or any(
            ord(character) <= 0x20 or ord(character) >= 0x7F
            for character in value
        )
        or "\\" in value
    ):
        raise RepositoryError(
            "Repository base URL must be bounded printable ASCII"
        )
    parsed = urlsplit(value)
    try:
        parsed.port
    except ValueError as error:
        raise RepositoryError(
            f"Repository base URL has an invalid port: {error}"
        ) from error
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RepositoryError(
            "Repository base URL must use HTTPS without credentials, "
            "query, or fragment"
        )
    if parsed.hostname == "github.com":
        segments = parsed.path.removeprefix("/").split("/")
        if (
            parsed.netloc != "github.com"
            or len(segments) != 5
            or GITHUB_SEGMENT.fullmatch(segments[0]) is None
            or GITHUB_SEGMENT.fullmatch(segments[1]) is None
            or segments[2:4] != ["releases", "download"]
            or GITHUB_RELEASE_TAG.fullmatch(segments[4]) is None
            or segments[4].lower() == "latest"
        ):
            raise RepositoryError(
                "GitHub repository base URL must be a canonical immutable "
                "release path"
            )
    return value.rstrip("/")


def _catalog_document(
    packs: Sequence[PackInput],
    base_url: str,
    catalog_id: str,
    generated_utc: str,
    expires_utc: str,
    minimum_client_version: str,
    catalog_key_id: str,
    catalog_private_key: Path,
    trusted: Mapping[str, TrustedKey],
    openssl: str,
    hosted_layout: str,
) -> dict[str, Any]:
    entries = []
    for pack in sorted(packs, key=lambda item: item.signed["pack_id"]):
        pack_id = pack.signed["pack_id"]
        hosted_prefix = (
            f"{base_url}/catalogs/manifests"
            if hosted_layout == "nested"
            else base_url
        )
        manifest_url = f"{hosted_prefix}/{pack_id}.json"
        archive_url = f"{hosted_prefix}/{pack.signed['archive']['file_name']}"
        if len(manifest_url) > 4096 or len(archive_url) > 4096:
            raise RepositoryError(
                f"Manifest or archive URL exceeds the runtime limit for {pack_id}"
            )
        entries.append(
            {
                "pack_id": pack_id,
                "manifest_url": manifest_url,
                "manifest_sha256": hashlib.sha256(
                    pack.manifest_bytes
                ).hexdigest(),
                "signing_key_id": pack.signing_key_id,
                "support_status": pack.signed["compatibility"]["support_status"],
            }
        )
    body = {
        "catalog_id": catalog_id,
        "generated_utc": generated_utc,
        "expires_utc": expires_utc,
        "minimum_client_version": minimum_client_version,
        "packs": entries,
    }
    signature = sign_with_openssl(
        canonical_json_bytes(body), catalog_private_key.resolve(), openssl
    )
    document = {
        "schema_version": 1,
        "kind": "cyxwiz-backend-pack-catalog",
        "signed": body,
        "signatures": [
            {
                "key_id": catalog_key_id,
                "algorithm": "ed25519",
                "value": base64.urlsafe_b64encode(signature)
                .decode("ascii")
                .rstrip("="),
            }
        ],
    }
    validate_catalog(document)
    _verified_envelope_key(
        document, trusted, "catalog", openssl, catalog_key_id
    )
    return document


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=2) + "\n").encode("utf-8")


def _publish_tree(
    output: Path,
    trust_bytes: bytes,
    catalog: Mapping[str, Any],
    packs: Sequence[PackInput],
    hosted_layout: str,
) -> None:
    output = output.resolve()
    if output.exists():
        raise RepositoryError(f"Output path already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent)
    )
    try:
        hosted = staging / "hosted"
        bootstrap = staging / "bootstrap"
        hosted_manifests = (
            hosted / "catalogs" / "manifests"
            if hosted_layout == "nested"
            else hosted
        )
        bootstrap_manifests = bootstrap / "catalogs" / "manifests"
        hosted_manifests.mkdir(parents=True)
        bootstrap_manifests.mkdir(parents=True)
        (bootstrap / "trust").mkdir(parents=True)
        (bootstrap / "trust" / "trusted-keys.json").write_bytes(trust_bytes)
        catalog_bytes = _json_bytes(catalog)
        hosted_catalog = (
            hosted / "catalogs" / "current.json"
            if hosted_layout == "nested"
            else hosted /
                f"cyxwiz-backend-catalog-{catalog['signed']['catalog_id']}.json"
        )
        hosted_catalog.parent.mkdir(parents=True, exist_ok=True)
        hosted_catalog.write_bytes(catalog_bytes)
        (bootstrap / "catalogs" / "current.json").write_bytes(catalog_bytes)
        for pack in packs:
            pack_id = pack.signed["pack_id"]
            (hosted_manifests / f"{pack_id}.json").write_bytes(
                pack.manifest_bytes
            )
            (bootstrap_manifests / f"{pack_id}.json").write_bytes(
                pack.manifest_bytes
            )
            shutil.copyfile(
                pack.source_archive,
                hosted_manifests / pack.signed["archive"]["file_name"],
            )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def prepare_repository(args: argparse.Namespace) -> str:
    trust_path = args.trust_root.resolve()
    trust_document, trust_bytes = _load_json(trust_path, "trust root")
    trusted = _trusted_keys(trust_document)
    catalog_key = trusted.get(args.catalog_key_id)
    if (
        catalog_key is None
        or catalog_key.revoked
        or "catalog" not in catalog_key.roles
    ):
        raise RepositoryError(
            "Catalog key ID is absent, revoked, or lacks the catalog role"
        )
    packs = [
        _load_pack(path, trusted, args.pack_key_id, args.openssl)
        for path in args.manifest
    ]
    _validate_repository_packs(packs)
    _validate_hosted_asset_names(
        packs, args.catalog_id, args.hosted_layout
    )
    base_url = validated_https_base_url(args.base_url)
    catalog = _catalog_document(
        packs,
        base_url,
        args.catalog_id,
        args.generated_utc,
        args.expires_utc,
        args.minimum_client_version,
        args.catalog_key_id,
        args.catalog_private_key,
        trusted,
        args.openssl,
        args.hosted_layout,
    )
    _publish_tree(
        args.output, trust_bytes, catalog, packs, args.hosted_layout
    )
    if args.hosted_layout == "flat":
        return f"{base_url}/cyxwiz-backend-catalog-{args.catalog_id}.json"
    return f"{base_url}/catalogs/current.json"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        catalog_url = prepare_repository(args)
    except (RepositoryError, ValueError, OSError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    print(f"[OK] Hosted repository: {args.output.resolve() / 'hosted'}")
    print(f"[OK] Bootstrap metadata: {args.output.resolve() / 'bootstrap'}")
    print(f"[OK] Catalog URL: {catalog_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
