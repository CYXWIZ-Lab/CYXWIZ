#!/usr/bin/env python3
"""Validate one complete, externally signed CyxWiz alpha upload directory."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from backend_pack_contract import (
    validate_catalog,
    validate_pack_manifest,
)
from installer_bundle_contract import validate_installer_bundle_descriptor
from prepare_backend_pack_repository import (
    GITHUB_RELEASE_TAG,
    RepositoryError,
    validated_https_base_url,
    verify_trusted_metadata_signature,
)
from prepare_installer_release_configuration import REPOSITORY, VERSION


INVENTORY_NAME = "release-inventory.json"
CHECKSUM_NAME = "SHA256SUMS.txt"
MAX_INVENTORY_BYTES = 2 * 1024 * 1024
MAX_ASSET_BYTES = 8 * 1024 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024 * 1024
MAX_ASSETS = 4096

_ASSET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,254}")
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SIGNATURE = re.compile(r"[A-Za-z0-9_-]{86}")

TARGETS = (
    ("windows", "x86_64", "cyxwiz-setup-windows-x64.zip"),
    ("linux", "x86_64", "cyxwiz-setup-ubuntu-x64.tar.gz"),
    ("macos", "x86_64", "cyxwiz-setup-macos-x64.tar.gz"),
    ("macos", "arm64", "cyxwiz-setup-macos-arm64.tar.gz"),
)
PACK_TARGETS = {
    ("win64", "x86_64"),
    ("linux64", "x86_64"),
    ("macos", "x86_64"),
    ("macos", "arm64"),
}


class AlphaPublicationError(ValueError):
    """Raised when a directory cannot be promoted as an installer alpha."""


def _exact_object(value: Any, label: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise AlphaPublicationError(f"{label} contains unknown or missing fields")
    return value


def _bounded_string(
    value: Any, label: str, pattern: re.Pattern[str], maximum: int = 255
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or pattern.fullmatch(value) is None
    ):
        raise AlphaPublicationError(f"{label} is invalid")
    return value


def _bounded_integer(value: Any, label: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AlphaPublicationError(f"{label} must be an integer")
    if value < 1 or value > maximum:
        raise AlphaPublicationError(f"{label} is outside its bounds")
    return value


def _load_json(path: Path, label: str, maximum: int = MAX_INVENTORY_BYTES) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AlphaPublicationError(f"{label} must be a regular non-link file")
    try:
        size = path.stat().st_size
        if size < 2 or size > maximum:
            raise AlphaPublicationError(f"{label} is outside its byte bound")
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AlphaPublicationError(f"cannot read {label}: {error}") from error
    if not isinstance(document, dict):
        raise AlphaPublicationError(f"{label} must contain a JSON object")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise AlphaPublicationError(f"cannot hash release asset {path.name}: {error}") from error
    return digest.hexdigest()


def _asset_name(value: Any, label: str) -> str:
    name = _bounded_string(value, label, _ASSET_NAME)
    if name in {INVENTORY_NAME, CHECKSUM_NAME} or name.endswith("."):
        raise AlphaPublicationError(f"{label} is reserved or non-portable")
    return name


def validate_inventory_document(document: Any) -> dict[str, Any]:
    envelope = _exact_object(
        document,
        "release inventory",
        {"schema_version", "kind", "signed", "signatures"},
    )
    if (
        type(envelope["schema_version"]) is not int
        or envelope["schema_version"] != 1
        or envelope["kind"] != "cyxwiz-alpha-release-inventory"
    ):
        raise AlphaPublicationError("release inventory identity is invalid")
    body = _exact_object(
        envelope["signed"],
        "signed release inventory",
        {
            "kind",
            "repository",
            "release_tag",
            "asset_base_url",
            "cyxwiz_release",
            "bundle_version",
            "catalog_url",
            "assets",
        },
    )
    if body["kind"] != "cyxwiz-alpha-release-assets":
        raise AlphaPublicationError("signed release inventory kind is invalid")
    _bounded_string(body["repository"], "repository", REPOSITORY)
    _bounded_string(body["release_tag"], "release tag", GITHUB_RELEASE_TAG, 128)
    _bounded_string(body["cyxwiz_release"], "CyxWiz release", VERSION, 64)
    _bounded_string(body["bundle_version"], "bundle version", VERSION, 64)
    try:
        body["asset_base_url"] = validated_https_base_url(body["asset_base_url"])
    except RepositoryError as error:
        raise AlphaPublicationError(str(error)) from error
    if not isinstance(body["catalog_url"], str) or len(body["catalog_url"]) > 4096:
        raise AlphaPublicationError("catalog URL is invalid")

    assets = body["assets"]
    if not isinstance(assets, list) or not 1 <= len(assets) <= MAX_ASSETS:
        raise AlphaPublicationError("release asset count is invalid")
    names: list[str] = []
    folded: set[str] = set()
    total = 0
    for index, raw in enumerate(assets):
        entry = _exact_object(raw, f"asset {index}", {"name", "size", "sha256"})
        name = _asset_name(entry["name"], f"asset {index} name")
        if name.casefold() in folded:
            raise AlphaPublicationError("release asset names collide case-insensitively")
        folded.add(name.casefold())
        names.append(name)
        total += _bounded_integer(entry["size"], f"asset {name} size", MAX_ASSET_BYTES)
        _bounded_string(entry["sha256"], f"asset {name} SHA-256", _SHA256, 64)
    if names != sorted(names):
        raise AlphaPublicationError("release asset inventory is not sorted")
    if total > MAX_TOTAL_BYTES:
        raise AlphaPublicationError("release asset inventory exceeds its total byte bound")

    signatures = envelope["signatures"]
    if not isinstance(signatures, list) or len(signatures) != 1:
        raise AlphaPublicationError("release inventory requires exactly one signature")
    signature = _exact_object(
        signatures[0], "release inventory signature", {"key_id", "algorithm", "value"}
    )
    _bounded_string(signature["key_id"], "signature key ID", _IDENTIFIER, 128)
    if signature["algorithm"] != "ed25519":
        raise AlphaPublicationError("release inventory signature algorithm is invalid")
    _bounded_string(signature["value"], "release inventory signature", _SIGNATURE, 86)
    return envelope


def build_signed_inventory(
    body: Mapping[str, Any], key_id: str, signature_value: str
) -> dict[str, Any]:
    document = {
        "schema_version": 1,
        "kind": "cyxwiz-alpha-release-inventory",
        "signed": dict(body),
        "signatures": [
            {"key_id": key_id, "algorithm": "ed25519", "value": signature_value}
        ],
    }
    validate_inventory_document(document)
    return document


def _require_expected_identity(
    body: Mapping[str, Any], repository: str, release_tag: str,
    cyxwiz_release: str, bundle_version: str, require_github: bool,
) -> None:
    expected = {
        "repository": repository,
        "release_tag": release_tag,
        "cyxwiz_release": cyxwiz_release,
        "bundle_version": bundle_version,
    }
    for field, value in expected.items():
        if body[field] != value:
            raise AlphaPublicationError(f"release inventory {field} does not match")
    if require_github:
        expected_base = (
            f"https://github.com/{repository}/releases/download/{release_tag}"
        )
        if body["asset_base_url"] != expected_base:
            raise AlphaPublicationError(
                "release inventory does not use the canonical GitHub Release root"
            )


def _require_metadata_url(base_url: str, url: Any, expected_name: str) -> None:
    expected = f"{base_url}/{expected_name}"
    if url != expected:
        raise AlphaPublicationError(f"metadata URL does not match asset {expected_name}")


def _verify_repository_assets(
    directory: Path, body: Mapping[str, Any], trust_root: Path, openssl: str
) -> set[str]:
    names = {entry["name"] for entry in body["assets"]}
    consumed: set[str] = set()
    catalog_name = urlsplit(body["catalog_url"]).path.rsplit("/", 1)[-1]
    if not catalog_name.startswith("cyxwiz-backend-catalog-") or catalog_name not in names:
        raise AlphaPublicationError("release inventory catalog asset is missing")
    _require_metadata_url(body["asset_base_url"], body["catalog_url"], catalog_name)
    catalog_path = directory / catalog_name
    consumed.add(catalog_name)
    catalog = _load_json(catalog_path, "backend-pack catalog")
    try:
        validate_catalog(catalog)
        verify_trusted_metadata_signature(catalog, trust_root, "catalog", openssl)
    except (ValueError, RepositoryError) as error:
        raise AlphaPublicationError(f"backend-pack catalog is not trusted: {error}") from error
    if catalog["signed"]["minimum_client_version"] != body["cyxwiz_release"]:
        raise AlphaPublicationError("catalog client version differs from the release")

    packs: list[Mapping[str, Any]] = []
    for entry in catalog["signed"]["packs"]:
        manifest_name = f"{entry['pack_id']}.json"
        if manifest_name not in names:
            raise AlphaPublicationError(f"pack manifest is missing: {manifest_name}")
        _require_metadata_url(
            body["asset_base_url"], entry["manifest_url"], manifest_name
        )
        manifest_path = directory / manifest_name
        consumed.add(manifest_name)
        if _sha256(manifest_path) != entry["manifest_sha256"]:
            raise AlphaPublicationError(
                f"catalog manifest SHA-256 differs for {manifest_name}"
            )
        manifest = _load_json(manifest_path, f"pack manifest {manifest_name}")
        try:
            validate_pack_manifest(manifest)
            verify_trusted_metadata_signature(
                manifest, trust_root, "pack", openssl, entry["signing_key_id"]
            )
        except (ValueError, RepositoryError) as error:
            raise AlphaPublicationError(
                f"pack manifest is not trusted: {manifest_name}: {error}"
            ) from error
        archive = manifest["signed"]["archive"]
        archive_name = archive["file_name"]
        if archive_name not in names:
            raise AlphaPublicationError(f"pack archive is missing: {archive_name}")
        archive_path = directory / archive_name
        consumed.add(archive_name)
        if (
            archive_path.stat().st_size != archive["size"]
            or _sha256(archive_path) != archive["sha256"]
        ):
            raise AlphaPublicationError(
                f"pack archive differs from its signed manifest: {archive_name}"
            )
        packs.append(manifest["signed"])

    for pack in packs:
        release = pack["cyxwiz_release"]
        if (
            release["minimum"] != body["cyxwiz_release"]
            or release["maximum"] != body["cyxwiz_release"]
        ):
            raise AlphaPublicationError(
                f"pack release range differs: {pack['pack_id']}"
            )

    for target in PACK_TARGETS:
        target_packs = [
            pack for pack in packs
            if (pack["platform"], pack["architecture"]) == target
        ]
        bases = {
            pack["pack_id"] for pack in target_packs if pack["pack_kind"] == "base"
        }
        optional = [
            pack for pack in target_packs
            if pack["pack_kind"] == "backend_pack"
            and pack["companion_base_id"] in bases
        ]
        if not bases or not optional:
            raise AlphaPublicationError(
                f"release repository lacks a complete pack matrix for {target}"
            )
    return consumed


def _verify_installer_assets(
    directory: Path, body: Mapping[str, Any], trust_root: Path, openssl: str
) -> set[str]:
    names = {entry["name"] for entry in body["assets"]}
    consumed: set[str] = set()
    for platform, architecture, setup_name in TARGETS:
        if setup_name not in names:
            raise AlphaPublicationError(f"setup package is missing: {setup_name}")
        consumed.add(setup_name)
        bundle_id = (
            f"cyxwiz-installer-{body['cyxwiz_release']}-{body['bundle_version']}-"
            f"{platform}-{architecture}"
        )
        descriptor_name = f"{bundle_id}.descriptor.json"
        archive_name = f"{bundle_id}.zip"
        if descriptor_name not in names or archive_name not in names:
            raise AlphaPublicationError(
                f"installer bundle pair is missing for {platform}-{architecture}"
            )
        consumed.update({descriptor_name, archive_name})
        descriptor = _load_json(
            directory / descriptor_name, f"installer descriptor {descriptor_name}"
        )
        try:
            validate_installer_bundle_descriptor(descriptor)
            verify_trusted_metadata_signature(
                descriptor, trust_root, "installer", openssl
            )
        except (ValueError, RepositoryError) as error:
            raise AlphaPublicationError(
                f"installer descriptor is not trusted: {descriptor_name}: {error}"
            ) from error
        signed = descriptor["signed"]
        if (
            signed["bundle_id"] != bundle_id
            or signed["cyxwiz_release"] != body["cyxwiz_release"]
            or signed["bundle_version"] != body["bundle_version"]
            or signed["release_channel"] != "alpha"
            or signed["platform"] != platform
            or signed["architecture"] != architecture
        ):
            raise AlphaPublicationError(
                f"installer descriptor identity differs: {descriptor_name}"
            )
        archive = signed["archive"]
        archive_path = directory / archive_name
        if (
            archive["file_name"] != archive_name
            or archive_path.stat().st_size != archive["size"]
            or _sha256(archive_path) != archive["sha256"]
        ):
            raise AlphaPublicationError(
                f"installer archive differs from its descriptor: {archive_name}"
            )
    return consumed


def validate_upload_directory(
    directory: Path,
    trust_root: Path,
    repository: str,
    release_tag: str,
    cyxwiz_release: str,
    bundle_version: str,
    *,
    openssl: str = "openssl",
    require_github: bool = False,
) -> dict[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise AlphaPublicationError("release upload root must be a direct directory")
    root = directory.resolve(strict=True)
    inventory_path = root / INVENTORY_NAME
    document = validate_inventory_document(
        _load_json(inventory_path, "release inventory")
    )
    body = document["signed"]
    _require_expected_identity(
        body, repository, release_tag, cyxwiz_release, bundle_version,
        require_github,
    )
    if trust_root.is_symlink() or not trust_root.is_file():
        raise AlphaPublicationError("trust root must be a regular non-link file")
    try:
        verify_trusted_metadata_signature(
            document, trust_root.resolve(strict=True), "installer", openssl
        )
    except (OSError, RepositoryError) as error:
        raise AlphaPublicationError(f"release inventory is not trusted: {error}") from error

    declared = {entry["name"]: entry for entry in body["assets"]}
    expected_names = set(declared) | {INVENTORY_NAME, CHECKSUM_NAME}
    observed: set[str] = set()
    folded: set[str] = set()
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise AlphaPublicationError(
                f"release upload contains a non-regular entry: {path.name}"
            )
        if path.name.casefold() in folded:
            raise AlphaPublicationError("release upload names collide case-insensitively")
        folded.add(path.name.casefold())
        observed.add(path.name)
    if observed != expected_names:
        missing = sorted(expected_names - observed)
        extra = sorted(observed - expected_names)
        raise AlphaPublicationError(
            f"release upload asset set differs (missing={missing}, extra={extra})"
        )
    for name, entry in declared.items():
        path = root / name
        if path.stat().st_size != entry["size"] or _sha256(path) != entry["sha256"]:
            raise AlphaPublicationError(f"release asset differs from inventory: {name}")

    expected_checksums = "".join(
        f"{declared[name]['sha256']}  {name}\n" for name in sorted(declared)
    ).encode("ascii")
    checksum_path = root / CHECKSUM_NAME
    if checksum_path.stat().st_size > MAX_INVENTORY_BYTES:
        raise AlphaPublicationError("checksum file exceeds its byte bound")
    if checksum_path.read_bytes() != expected_checksums:
        raise AlphaPublicationError("checksum file is not canonical or complete")

    consumed = _verify_repository_assets(root, body, trust_root, openssl)
    consumed.update(_verify_installer_assets(root, body, trust_root, openssl))
    if consumed != set(declared):
        raise AlphaPublicationError(
            "signed inventory contains an asset outside the release contracts"
        )
    return document
