"""Schema-1 contract for a signed CyxWiz installer bootstrap bundle."""

from __future__ import annotations

import base64
import re
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any

from backend_pack_contract import canonical_json_bytes


class InstallerBundleContractError(ValueError):
    """Raised when an installer-bundle descriptor violates the contract."""


MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_COMPONENT_BYTES = 512 * 1024 * 1024
MAX_COMPONENTS = 4096

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SIGNATURE = re.compile(r"^[A-Za-z0-9_-]{86}$")


def _exact_keys(value: Any, label: str, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise InstallerBundleContractError(
            f"{label} contains unknown or missing fields"
        )
    return value


def _string(value: Any, label: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise InstallerBundleContractError(f"{label} is invalid")
    return value


def _unsigned(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InstallerBundleContractError(f"{label} must be an integer")
    if value < minimum or value > maximum:
        raise InstallerBundleContractError(f"{label} is outside its bounds")
    return value


def _utc(value: Any, label: str) -> str:
    text = _string(value, label, _UTC)
    try:
        datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise InstallerBundleContractError(f"{label} is invalid") from error
    return text


def _component_path(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise InstallerBundleContractError("component path is invalid")
    if "\\" in value or ":" in value or value.startswith("/"):
        raise InstallerBundleContractError("component path is not portable")
    path = PurePosixPath(value)
    if str(path) != value or any(part in {"", ".", ".."} for part in path.parts):
        raise InstallerBundleContractError("component path is not canonical")
    return value


def validate_installer_bundle_descriptor(
    document: Any, *, require_signature: bool = True
) -> dict[str, Any]:
    envelope = _exact_keys(
        document,
        "descriptor",
        {"schema_version", "kind", "signed", "signatures"},
    )
    if (
        type(envelope["schema_version"]) is not int
        or envelope["schema_version"] != 1
        or envelope["kind"] != "cyxwiz-installer-bundle"
    ):
        raise InstallerBundleContractError("descriptor identity is invalid")

    body = _exact_keys(
        envelope["signed"],
        "signed body",
        {
            "bundle_id",
            "bundle_version",
            "cyxwiz_release",
            "release_channel",
            "platform",
            "architecture",
            "minimum_setup_version",
            "generated_utc",
            "expires_utc",
            "archive",
            "components",
        },
    )
    bundle_id = _string(body["bundle_id"], "bundle_id", _IDENTIFIER)
    _string(body["bundle_version"], "bundle_version", _VERSION)
    _string(body["cyxwiz_release"], "cyxwiz_release", _VERSION)
    if body["release_channel"] not in {"alpha", "beta", "stable"}:
        raise InstallerBundleContractError("release_channel is invalid")
    if body["platform"] not in {"windows", "linux", "macos"}:
        raise InstallerBundleContractError("platform is invalid")
    if body["architecture"] not in {"x86_64", "arm64"}:
        raise InstallerBundleContractError("architecture is invalid")
    _string(body["minimum_setup_version"], "minimum_setup_version", _VERSION)
    generated = _utc(body["generated_utc"], "generated_utc")
    expires = _utc(body["expires_utc"], "expires_utc")
    if expires <= generated:
        raise InstallerBundleContractError("descriptor expiry must follow generation")

    archive = _exact_keys(
        body["archive"], "archive", {"file_name", "size", "sha256"}
    )
    if archive["file_name"] != f"{bundle_id}.zip":
        raise InstallerBundleContractError("archive file name does not match bundle_id")
    _unsigned(archive["size"], "archive size", 1, MAX_ARCHIVE_BYTES)
    _string(archive["sha256"], "archive sha256", _SHA256)

    components = body["components"]
    if (
        not isinstance(components, list)
        or not 1 <= len(components) <= MAX_COMPONENTS
    ):
        raise InstallerBundleContractError("component inventory size is invalid")
    seen: set[str] = set()
    total = 0
    paths: set[str] = set()
    ordered_paths: list[str] = []
    executable_paths: set[str] = set()
    for item in components:
        component = _exact_keys(
            item, "component", {"path", "size", "sha256", "executable"}
        )
        path = _component_path(component["path"])
        folded = path.casefold()
        if folded in seen:
            raise InstallerBundleContractError("component paths are duplicated")
        seen.add(folded)
        paths.add(path)
        ordered_paths.append(path)
        total += _unsigned(component["size"], "component size", 0, MAX_COMPONENT_BYTES)
        _string(component["sha256"], "component sha256", _SHA256)
        if not isinstance(component["executable"], bool):
            raise InstallerBundleContractError("component executable must be boolean")
        if component["executable"]:
            executable_paths.add(path)
    if ordered_paths != sorted(ordered_paths):
        raise InstallerBundleContractError("component inventory is not canonical")
    if total > MAX_COMPONENT_BYTES:
        raise InstallerBundleContractError("component inventory exceeds its byte budget")

    executable_suffix = ".exe" if body["platform"] == "windows" else ""
    required_executables = {
        f"cyxwiz-installer{executable_suffix}",
        f"cyxwiz-backend-pack-installer{executable_suffix}",
    }
    required_metadata = {
        "runtime/trust/trusted-keys.json",
        "runtime/catalogs/current.json",
    }
    if not (required_executables | required_metadata).issubset(paths):
        raise InstallerBundleContractError("installer bundle is missing required bootstrap files")
    if not required_executables.issubset(executable_paths):
        raise InstallerBundleContractError("installer entry points are not executable")

    signatures = envelope["signatures"]
    if not isinstance(signatures, list) or (require_signature and not signatures):
        raise InstallerBundleContractError("descriptor requires a signature")
    signature_ids: set[str] = set()
    for item in signatures:
        signature = _exact_keys(
            item, "signature", {"key_id", "algorithm", "value"}
        )
        key_id = _string(signature["key_id"], "signature key_id", _IDENTIFIER)
        if key_id in signature_ids or signature["algorithm"] != "ed25519":
            raise InstallerBundleContractError("signature identity is invalid")
        signature_ids.add(key_id)
        if not isinstance(signature["value"], str) or not _SIGNATURE.fullmatch(
            signature["value"]
        ):
            raise InstallerBundleContractError("signature value is invalid")
        try:
            decoded = base64.urlsafe_b64decode(signature["value"] + "==")
        except (TypeError, ValueError) as error:
            raise InstallerBundleContractError("signature value is invalid") from error
        if len(decoded) != 64:
            raise InstallerBundleContractError("signature value is invalid")
    return envelope
