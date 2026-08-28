#!/usr/bin/env python3
"""Validate and materialize public setup configuration for a release build."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backend_pack_contract import validate_trust_root  # noqa: E402
from prepare_backend_pack_repository import (  # noqa: E402
    RepositoryError,
    direct_https_base_url,
)


MAX_TRUST_BYTES = 4 * 1024 * 1024
RELEASE_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.-]+)?")
TARGETS = {
    ("windows", "x86_64"),
    ("linux", "x86_64"),
    ("macos", "x86_64"),
    ("macos", "arm64"),
}


class ReleaseConfigurationError(RuntimeError):
    """Raised when setup release configuration is not publishable."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trust-store", required=True, type=Path)
    parser.add_argument(
        "--asset-base-url",
        required=True,
        help="Direct non-redirecting HTTPS root containing release assets",
    )
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--cyxwiz-release", required=True)
    parser.add_argument("--bundle-version", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--trust-output", required=True, type=Path)
    parser.add_argument("--configuration-output", required=True, type=Path)
    return parser.parse_args(argv)


def _read_trust(path: Path) -> tuple[bytes, dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseConfigurationError("trust store must be a regular file")
    size = path.stat().st_size
    if size <= 0 or size > MAX_TRUST_BYTES:
        raise ReleaseConfigurationError(
            f"trust store must contain 1-{MAX_TRUST_BYTES} bytes"
        )
    content = path.read_bytes()
    try:
        document = json.loads(content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseConfigurationError(
            f"trust store is not valid UTF-8 JSON: {error}"
        ) from error
    validate_trust_root(document)
    if not any(
        not key["revoked"] and "installer" in key["roles"]
        for key in document["keys"]
    ):
        raise ReleaseConfigurationError(
            "trust store requires an active installer signing key"
        )
    return content, document


def versioned_asset_base_url(value: str, release_tag: str) -> str:
    if RELEASE_TAG.fullmatch(release_tag) is None:
        raise ReleaseConfigurationError("release tag is not a safe asset tag")
    try:
        base_url = direct_https_base_url(value)
    except RepositoryError as error:
        raise ReleaseConfigurationError(str(error)) from error
    if urlsplit(base_url).path.rsplit("/", 1)[-1] != release_tag:
        raise ReleaseConfigurationError(
            "asset base URL must end with the exact immutable release tag"
        )
    return base_url


def _validate_identity(args: argparse.Namespace) -> None:
    args.asset_base_url = versioned_asset_base_url(
        args.asset_base_url, args.release_tag
    )
    for label, value in (
        ("CyxWiz release", args.cyxwiz_release),
        ("bundle version", args.bundle_version),
    ):
        if VERSION.fullmatch(value) is None:
            raise ReleaseConfigurationError(f"{label} is invalid")
    if (args.platform, args.architecture) not in TARGETS:
        raise ReleaseConfigurationError("platform/architecture is unsupported")


def prepare(args: argparse.Namespace) -> dict[str, str]:
    _validate_identity(args)
    trust_bytes, _ = _read_trust(args.trust_store.resolve(strict=True))
    descriptor_name = (
        f"cyxwiz-installer-{args.cyxwiz_release}-{args.bundle_version}-"
        f"{args.platform}-{args.architecture}.descriptor.json"
    ).lower()
    descriptor_url = (
        f"{args.asset_base_url}/{descriptor_name}"
    )
    trust_output = args.trust_output.resolve()
    configuration_output = args.configuration_output.resolve()
    if trust_output == configuration_output:
        raise ReleaseConfigurationError("configuration outputs must be distinct")
    trust_output.parent.mkdir(parents=True, exist_ok=True)
    configuration_output.parent.mkdir(parents=True, exist_ok=True)
    configuration = {
        "descriptor_url": descriptor_url,
        "trust_sha256": hashlib.sha256(trust_bytes).hexdigest(),
    }
    with tempfile.TemporaryDirectory(
        prefix="cyxwiz-release-configuration-",
        dir=configuration_output.parent,
    ) as temporary_name:
        temporary = Path(temporary_name)
        prepared_trust = temporary / "trusted-keys.json"
        prepared_configuration = temporary / "configuration.json"
        prepared_trust.write_bytes(trust_bytes)
        prepared_configuration.write_text(
            json.dumps(configuration, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        shutil.copyfile(prepared_trust, trust_output)
        shutil.copyfile(prepared_configuration, configuration_output)
    return configuration


def main(argv: Sequence[str] | None = None) -> int:
    try:
        configuration = prepare(parse_args(argv))
    except (OSError, ValueError, ReleaseConfigurationError) as error:
        print(f"[ERROR] {error}")
        return 1
    print(f"[OK] Installer descriptor URL: {configuration['descriptor_url']}")
    print(f"[OK] Embedded trust SHA-256: {configuration['trust_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
