#!/usr/bin/env python3
"""Create a deterministic, signable CyxWiz installer bootstrap bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Sequence

from installer_bundle_contract import (
    MAX_COMPONENTS,
    canonical_json_bytes,
    validate_installer_bundle_descriptor,
)


class InstallerBundlePackagingError(RuntimeError):
    """Raised when a staged installer cannot become a safe bundle."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bundle-version", required=True)
    parser.add_argument("--cyxwiz-release", required=True)
    parser.add_argument("--release-channel", choices=("alpha", "beta", "stable"), required=True)
    parser.add_argument("--platform", choices=("windows", "linux", "macos"), required=True)
    parser.add_argument("--architecture", choices=("x86_64", "arm64"), required=True)
    parser.add_argument("--minimum-setup-version", required=True)
    parser.add_argument("--generated-utc", required=True)
    parser.add_argument("--expires-utc", required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, content: bytes) -> None:
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
        temporary.unlink(missing_ok=True)


def staged_files(stage: Path) -> list[Path]:
    if not stage.is_absolute() or not stage.is_dir() or stage.is_symlink():
        raise InstallerBundlePackagingError("stage must be an exact absolute directory")
    files: list[Path] = []
    for entry in stage.rglob("*"):
        if entry.is_symlink():
            raise InstallerBundlePackagingError(f"stage contains a link: {entry}")
        if entry.is_file():
            files.append(entry)
        elif not entry.is_dir():
            raise InstallerBundlePackagingError(f"stage contains an unsupported entry: {entry}")
    if not files or len(files) > MAX_COMPONENTS:
        raise InstallerBundlePackagingError("stage file count is outside its bounds")
    return sorted(files, key=lambda item: item.relative_to(stage).as_posix())


def create_bundle(stage: Path, destination: Path, files: list[Path]) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for path in files:
                relative = path.relative_to(stage).as_posix()
                executable = bool(path.stat().st_mode & stat.S_IXUSR)
                info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                permissions = 0o755 if executable else 0o644
                info.external_attr = (stat.S_IFREG | permissions) << 16
                archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def component_inventory(stage: Path, files: list[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(stage).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
            "executable": bool(path.stat().st_mode & stat.S_IXUSR),
        }
        for path in files
    ]


def run(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    stage = args.stage.resolve(strict=True)
    output = args.output.resolve()
    if output == stage or stage in output.parents:
        raise InstallerBundlePackagingError("output must be outside the staged installer")
    output.mkdir(parents=True, exist_ok=True)
    bundle_id = (
        f"cyxwiz-installer-{args.cyxwiz_release}-{args.bundle_version}-"
        f"{args.platform}-{args.architecture}"
    ).lower()
    archive = output / f"{bundle_id}.zip"
    descriptor_path = output / f"{bundle_id}.descriptor.json"
    signature_input = output / f"{bundle_id}.signed.json"
    published = (archive, descriptor_path, signature_input)
    if any(path.exists() for path in published):
        raise InstallerBundlePackagingError(
            "versioned installer-bundle output already exists"
        )
    files = staged_files(stage)
    inventory = component_inventory(stage, files)
    with tempfile.TemporaryDirectory(
        prefix="cyxwiz-installer-bundle-", dir=output
    ) as temporary_name:
        temporary = Path(temporary_name)
        prepared_archive = temporary / archive.name
        create_bundle(stage, prepared_archive, files)
        if component_inventory(stage, files) != inventory:
            raise InstallerBundlePackagingError(
                "staged installer changed while the bundle was being created"
            )
        body = {
            "bundle_id": bundle_id,
            "bundle_version": args.bundle_version,
            "cyxwiz_release": args.cyxwiz_release,
            "release_channel": args.release_channel,
            "platform": args.platform,
            "architecture": args.architecture,
            "minimum_setup_version": args.minimum_setup_version,
            "generated_utc": args.generated_utc,
            "expires_utc": args.expires_utc,
            "archive": {
                "file_name": archive.name,
                "size": prepared_archive.stat().st_size,
                "sha256": sha256_file(prepared_archive),
            },
            "components": inventory,
        }
        descriptor = {
            "schema_version": 1,
            "kind": "cyxwiz-installer-bundle",
            "signed": body,
            "signatures": [],
        }
        validate_installer_bundle_descriptor(descriptor, require_signature=False)
        prepared_descriptor = temporary / descriptor_path.name
        prepared_signature_input = temporary / signature_input.name
        prepared_descriptor.write_text(
            json.dumps(descriptor, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        prepared_signature_input.write_bytes(canonical_json_bytes(body))
        os.replace(prepared_archive, archive)
        os.replace(prepared_signature_input, signature_input)
        os.replace(prepared_descriptor, descriptor_path)
    return archive, descriptor_path, signature_input


def main(argv: Sequence[str] | None = None) -> int:
    try:
        outputs = run(parse_args(argv))
    except (InstallerBundlePackagingError, OSError, ValueError) as error:
        print(f"[ERROR] {error}")
        return 1
    for output in outputs:
        print(f"[OK] {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
