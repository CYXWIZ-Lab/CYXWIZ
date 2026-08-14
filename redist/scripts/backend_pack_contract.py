#!/usr/bin/env python3
"""Strict schema-1 contracts for signed CyxWiz backend-pack metadata."""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
SIGNATURE_ALGORITHM = "ed25519"
BACKENDS = ("cpu", "cuda", "opencl", "oneapi")
PACK_KINDS = ("base", "backend_pack")
SUPPORT_STATES = ("supported", "diagnostic", "blocked", "revoked")
TRUST_ROLES = ("catalog", "pack")
IDENTITY_CONFIDENCE = (
    "unknown",
    "backend_local",
    "provider_reported",
    "stable_hardware",
)

_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_VERSION = re.compile(r"[0-9A-Za-z][0-9A-Za-z._+-]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
_ARCHITECTURES = ("x86_64", "arm64")
_PLATFORMS = ("win64", "linux64", "macos")


class ContractError(ValueError):
    """Raised when release metadata violates the frozen schema-1 contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON for detached signature input."""
    _reject_floats(value, "document")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _reject_floats(value: Any, field: str) -> None:
    if isinstance(value, float):
        raise ContractError(f"{field} must not contain floating-point values")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_floats(item, f"{field}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_floats(item, f"{field}[{index}]")


def _object(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{field} must be an object")
    return value


def _list(value: Any, field: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{field} must be an array")
    return value


def _exact_keys(
    value: Mapping[str, Any], field: str, required: set[str], optional: set[str] = set()
) -> None:
    missing = required.difference(value)
    unexpected = set(value).difference(required | optional)
    if missing:
        raise ContractError(f"{field} is missing: {', '.join(sorted(missing))}")
    if unexpected:
        raise ContractError(
            f"{field} contains unsupported fields: {', '.join(sorted(unexpected))}"
        )


def _string(value: Any, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ContractError(f"{field} must be a non-empty string")
    return value


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{field} must be an integer >= {minimum}")
    return value


def _identifier(value: Any, field: str) -> str:
    text = _string(value, field)
    if not _ID.fullmatch(text):
        raise ContractError(f"{field} is not a canonical identifier")
    return text


def _version(value: Any, field: str) -> str:
    text = _string(value, field)
    if not _VERSION.fullmatch(text):
        raise ContractError(f"{field} is not a valid version")
    return text


def _sha256(value: Any, field: str) -> str:
    text = _string(value, field)
    if not _SHA256.fullmatch(text):
        raise ContractError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _utc(value: Any, field: str) -> str:
    text = _string(value, field)
    if not _UTC.fullmatch(text):
        raise ContractError(f"{field} must use UTC YYYY-MM-DDTHH:MM:SSZ")
    return text


def _relative_path(value: Any, field: str) -> str:
    text = _string(value, field)
    if "\\" in text or text.startswith("/") or re.match(r"^[A-Za-z]:", text):
        raise ContractError(f"{field} must be a canonical relative POSIX path")
    path = PurePosixPath(text)
    if (
        path.as_posix() != text
        or any(part in ("", ".", "..") for part in path.parts)
    ):
        raise ContractError(f"{field} contains an unsafe path segment")
    return text


def _string_map(value: Any, field: str) -> None:
    entries = _object(value, field)
    for key, item in entries.items():
        _identifier(key, f"{field} key")
        _string(item, f"{field}.{key}")


def _validate_signatures(value: Any) -> None:
    signatures = _list(value, "signatures")
    if not signatures:
        raise ContractError("signatures must contain at least one trusted signature")
    seen: set[str] = set()
    for index, raw in enumerate(signatures):
        field = f"signatures[{index}]"
        signature = _object(raw, field)
        _exact_keys(signature, field, {"key_id", "algorithm", "value"})
        key_id = _identifier(signature["key_id"], f"{field}.key_id")
        if key_id in seen:
            raise ContractError(f"duplicate signature key_id: {key_id}")
        seen.add(key_id)
        if signature["algorithm"] != SIGNATURE_ALGORITHM:
            raise ContractError(f"{field}.algorithm must be {SIGNATURE_ALGORITHM}")
        encoded = _string(signature["value"], f"{field}.value")
        if not re.fullmatch(r"[A-Za-z0-9_-]{86}", encoded):
            raise ContractError(f"{field}.value must be unpadded base64url Ed25519")


def _validate_envelope(document: Any, kind: str) -> Mapping[str, Any]:
    envelope = _object(document, "document")
    _exact_keys(envelope, "document", {"schema_version", "kind", "signed", "signatures"})
    if envelope["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"schema_version must be {SCHEMA_VERSION}")
    if envelope["kind"] != kind:
        raise ContractError(f"kind must be {kind}")
    _validate_signatures(envelope["signatures"])
    signed = _object(envelope["signed"], "signed")
    canonical_json_bytes(signed)
    return signed


def _validate_components(value: Any) -> set[str]:
    components = _list(value, "signed.components")
    if not components:
        raise ContractError("signed.components must not be empty")
    paths: set[str] = set()
    folded: set[str] = set()
    for index, raw in enumerate(components):
        field = f"signed.components[{index}]"
        component = _object(raw, field)
        _exact_keys(component, field, {"path", "size", "sha256", "source", "executable"})
        path = _relative_path(component["path"], f"{field}.path")
        if path.casefold() in folded:
            raise ContractError(f"duplicate canonical component path: {path}")
        paths.add(path)
        folded.add(path.casefold())
        _integer(component["size"], f"{field}.size")
        _sha256(component["sha256"], f"{field}.sha256")
        _identifier(component["source"], f"{field}.source")
        if not isinstance(component["executable"], bool):
            raise ContractError(f"{field}.executable must be boolean")
    return paths


def validate_pack_manifest(document: Any) -> None:
    signed = _validate_envelope(document, "cyxwiz-backend-pack-manifest")
    required = {
        "pack_id", "pack_kind", "backend", "package_version", "platform",
        "architecture", "runtime_set_id", "cyxwiz_release", "arrayfire",
        "companion_base_id", "conflicts", "compatibility", "components",
        "licenses", "archive", "generated_utc",
    }
    _exact_keys(signed, "signed", required)
    pack_id = _identifier(signed["pack_id"], "signed.pack_id")
    pack_kind = signed["pack_kind"]
    if pack_kind not in PACK_KINDS:
        raise ContractError(f"signed.pack_kind must be one of {PACK_KINDS}")
    backend = signed["backend"]
    if backend not in BACKENDS:
        raise ContractError(f"signed.backend must be one of {BACKENDS}")
    if (pack_kind == "base") != (backend == "cpu"):
        raise ContractError("the base pack must be CPU and CPU must not be optional")
    _version(signed["package_version"], "signed.package_version")
    if signed["platform"] not in _PLATFORMS:
        raise ContractError(f"signed.platform must be one of {_PLATFORMS}")
    if signed["architecture"] not in _ARCHITECTURES:
        raise ContractError(f"signed.architecture must be one of {_ARCHITECTURES}")
    _identifier(signed["runtime_set_id"], "signed.runtime_set_id")
    _utc(signed["generated_utc"], "signed.generated_utc")

    release = _object(signed["cyxwiz_release"], "signed.cyxwiz_release")
    _exact_keys(release, "signed.cyxwiz_release", {"minimum", "maximum"})
    _version(release["minimum"], "signed.cyxwiz_release.minimum")
    _version(release["maximum"], "signed.cyxwiz_release.maximum")

    arrayfire = _object(signed["arrayfire"], "signed.arrayfire")
    _exact_keys(arrayfire, "signed.arrayfire", {"version", "abi"})
    _version(arrayfire["version"], "signed.arrayfire.version")
    _identifier(arrayfire["abi"], "signed.arrayfire.abi")

    companion = signed["companion_base_id"]
    if pack_kind == "base":
        if companion is not None:
            raise ContractError("base pack companion_base_id must be null")
    else:
        _identifier(companion, "signed.companion_base_id")
        if companion == pack_id:
            raise ContractError("a pack cannot require itself as its base")

    conflicts = _list(signed["conflicts"], "signed.conflicts")
    conflict_ids = [_identifier(item, "signed.conflicts[]") for item in conflicts]
    if len(conflict_ids) != len(set(conflict_ids)) or pack_id in conflict_ids:
        raise ContractError("signed.conflicts must be unique and exclude this pack")

    compatibility = _object(signed["compatibility"], "signed.compatibility")
    _exact_keys(
        compatibility,
        "signed.compatibility",
        {
            "device_kinds", "cpu_features", "provider_types",
            "minimum_driver_versions", "tested_driver_ranges",
            "minimum_identity_confidence", "recommendation_targets",
            "operation_matrix_id", "training_scope", "support_status",
        },
    )
    for name in ("device_kinds", "cpu_features", "provider_types", "training_scope"):
        values = _list(compatibility[name], f"signed.compatibility.{name}")
        identifiers = [
            _identifier(item, f"signed.compatibility.{name}[]")
            for item in values
        ]
        if len(identifiers) != len(set(identifiers)):
            raise ContractError(f"signed.compatibility.{name} must be unique")
    _string_map(
        compatibility["minimum_driver_versions"],
        "signed.compatibility.minimum_driver_versions",
    )
    _string_map(
        compatibility["tested_driver_ranges"],
        "signed.compatibility.tested_driver_ranges",
    )
    if compatibility["minimum_identity_confidence"] not in IDENTITY_CONFIDENCE:
        raise ContractError("unsupported minimum_identity_confidence")
    recommendation_targets = _list(
        compatibility["recommendation_targets"],
        "signed.compatibility.recommendation_targets",
    )
    if any(item not in BACKENDS for item in recommendation_targets):
        raise ContractError("recommendation_targets contains an unknown backend")
    if len(recommendation_targets) != len(set(recommendation_targets)):
        raise ContractError("recommendation_targets must be unique")
    _identifier(
        compatibility["operation_matrix_id"],
        "signed.compatibility.operation_matrix_id",
    )
    if compatibility["support_status"] not in SUPPORT_STATES:
        raise ContractError(f"unsupported support_status for {pack_id}")

    component_paths = _validate_components(signed["components"])
    licenses = _list(signed["licenses"], "signed.licenses")
    if not licenses:
        raise ContractError("signed.licenses must not be empty")
    for index, raw in enumerate(licenses):
        field = f"signed.licenses[{index}]"
        license_entry = _object(raw, field)
        _exact_keys(license_entry, field, {"component", "path"})
        _identifier(license_entry["component"], f"{field}.component")
        path = _relative_path(license_entry["path"], f"{field}.path")
        if path not in component_paths:
            raise ContractError(f"{field}.path is not a packaged component")

    archive = _object(signed["archive"], "signed.archive")
    _exact_keys(archive, "signed.archive", {"file_name", "size", "sha256"})
    file_name = _relative_path(archive["file_name"], "signed.archive.file_name")
    if "/" in file_name:
        raise ContractError("signed.archive.file_name must not contain directories")
    _integer(archive["size"], "signed.archive.size", minimum=1)
    _sha256(archive["sha256"], "signed.archive.sha256")


def validate_catalog(document: Any) -> None:
    signed = _validate_envelope(document, "cyxwiz-backend-pack-catalog")
    _exact_keys(
        signed,
        "signed",
        {"catalog_id", "generated_utc", "expires_utc", "minimum_client_version", "packs"},
    )
    _identifier(signed["catalog_id"], "signed.catalog_id")
    generated = _utc(signed["generated_utc"], "signed.generated_utc")
    expires = _utc(signed["expires_utc"], "signed.expires_utc")
    if expires <= generated:
        raise ContractError("signed.expires_utc must be later than generated_utc")
    _version(signed["minimum_client_version"], "signed.minimum_client_version")
    packs = _list(signed["packs"], "signed.packs")
    seen: set[str] = set()
    for index, raw in enumerate(packs):
        field = f"signed.packs[{index}]"
        entry = _object(raw, field)
        _exact_keys(
            entry,
            field,
            {"pack_id", "manifest_url", "manifest_sha256", "signing_key_id", "support_status"},
        )
        pack_id = _identifier(entry["pack_id"], f"{field}.pack_id")
        if pack_id in seen:
            raise ContractError(f"duplicate catalog pack_id: {pack_id}")
        seen.add(pack_id)
        url = _string(entry["manifest_url"], f"{field}.manifest_url")
        if not url.startswith("https://"):
            raise ContractError(f"{field}.manifest_url must use HTTPS")
        _sha256(entry["manifest_sha256"], f"{field}.manifest_sha256")
        _identifier(entry["signing_key_id"], f"{field}.signing_key_id")
        if entry["support_status"] not in SUPPORT_STATES:
            raise ContractError(f"unsupported catalog support status for {pack_id}")


def validate_trust_root(document: Any) -> None:
    """Validate the app-bundled public-key file consumed by native runtime code."""
    root = _object(document, "document")
    _exact_keys(root, "document", {"schema_version", "keys"})
    if root["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"schema_version must be {SCHEMA_VERSION}")
    keys = _list(root["keys"], "keys")
    if not keys:
        raise ContractError("keys must not be empty")
    seen: set[str] = set()
    for index, raw in enumerate(keys):
        field = f"keys[{index}]"
        key = _object(raw, field)
        _exact_keys(
            key,
            field,
            {"key_id", "algorithm", "public_key", "roles", "revoked"},
        )
        key_id = _identifier(key["key_id"], f"{field}.key_id")
        if key_id in seen:
            raise ContractError(f"duplicate trust key_id: {key_id}")
        seen.add(key_id)
        if key["algorithm"] != SIGNATURE_ALGORITHM:
            raise ContractError(f"{field}.algorithm must be {SIGNATURE_ALGORITHM}")
        encoded = _string(key["public_key"], f"{field}.public_key")
        if not re.fullmatch(r"[A-Za-z0-9_-]{43}", encoded):
            raise ContractError(
                f"{field}.public_key must be an unpadded base64url Ed25519 key"
            )
        roles = _list(key["roles"], f"{field}.roles")
        if not roles or any(role not in TRUST_ROLES for role in roles):
            raise ContractError(f"{field}.roles contains an unsupported role")
        if len(roles) != len(set(roles)):
            raise ContractError(f"{field}.roles must be unique")
        if not isinstance(key["revoked"], bool):
            raise ContractError(f"{field}.revoked must be boolean")


def validate_active_runtime(document: Any) -> None:
    runtime = _object(document, "document")
    _exact_keys(
        runtime,
        "document",
        {"schema_version", "runtime_set_id", "generation", "base_pack_id", "packs"},
    )
    if runtime["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"schema_version must be {SCHEMA_VERSION}")
    _identifier(runtime["runtime_set_id"], "runtime_set_id")
    _integer(runtime["generation"], "generation", minimum=1)
    _identifier(runtime["base_pack_id"], "base_pack_id")
    packs = _list(runtime["packs"], "packs")
    seen: set[str] = set()
    for index, raw in enumerate(packs):
        field = f"packs[{index}]"
        entry = _object(raw, field)
        _exact_keys(entry, field, {"backend", "pack_id"})
        backend = entry["backend"]
        if backend not in BACKENDS or backend == "cpu":
            raise ContractError(f"{field}.backend must be an optional backend")
        if backend in seen:
            raise ContractError(f"duplicate active backend: {backend}")
        seen.add(backend)
        _identifier(entry["pack_id"], f"{field}.pack_id")


def validate_runtime_composition(base_document: Any, pack_document: Any) -> None:
    """Reject base/plugin mixing before a runtime set can be activated."""
    validate_pack_manifest(base_document)
    validate_pack_manifest(pack_document)
    base = _object(base_document["signed"], "base.signed")
    pack = _object(pack_document["signed"], "pack.signed")
    if base["pack_kind"] != "base":
        raise ContractError("runtime composition base document is not a base pack")
    if pack["pack_kind"] != "backend_pack":
        raise ContractError("runtime composition optional document is not a backend pack")
    if pack["companion_base_id"] != base["pack_id"]:
        raise ContractError("backend pack requires a different companion base")
    if pack["runtime_set_id"] != base["runtime_set_id"]:
        raise ContractError("backend pack and base use different runtime sets")
    if pack["platform"] != base["platform"] or pack["architecture"] != base["architecture"]:
        raise ContractError("backend pack and base target different platforms")
    if pack["arrayfire"] != base["arrayfire"]:
        raise ContractError("backend pack and base have incompatible ArrayFire ABI")
