#!/usr/bin/env python3
"""Assemble signed CyxWiz alpha assets without uploading or creating a tag."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import package_installer_bundle as installer_packager  # noqa: E402
import prepare_backend_pack_repository as pack_repository  # noqa: E402
from backend_pack_contract import validate_pack_manifest  # noqa: E402
from prepare_installer_release_configuration import (  # noqa: E402
    REPOSITORY,
    ReleaseConfigurationError,
    VERSION,
    versioned_asset_base_url,
)
from sign_installer_bundle import sign_descriptor  # noqa: E402


@dataclass(frozen=True)
class ReleaseTarget:
    key: str
    platform: str
    architecture: str
    pack_platform: str
    setup_name: str


TARGETS = (
    ReleaseTarget(
        "windows-x64", "windows", "x86_64", "win64",
        "cyxwiz-setup-windows-x64.zip",
    ),
    ReleaseTarget(
        "linux-x64", "linux", "x86_64", "linux64",
        "cyxwiz-setup-ubuntu-x64.tar.gz",
    ),
    ReleaseTarget(
        "macos-x64", "macos", "x86_64", "macos",
        "cyxwiz-setup-macos-x64.tar.gz",
    ),
    ReleaseTarget(
        "macos-arm64", "macos", "arm64", "macos",
        "cyxwiz-setup-macos-arm64.tar.gz",
    ),
)
TARGET_BY_KEY = {target.key: target for target in TARGETS}
TARGET_BY_PACK = {
    (target.pack_platform, target.architecture): target for target in TARGETS
}


class AlphaReleaseError(RuntimeError):
    """Raised when inputs cannot form one complete alpha release."""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--installer-stage", action="append", required=True,
        help="TARGET=PATH; required once for every supported target",
    )
    parser.add_argument(
        "--setup-package", action="append", required=True,
        help="TARGET=PATH; required once for every supported target",
    )
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--trust-root", required=True, type=Path)
    parser.add_argument("--catalog-private-key", required=True, type=Path)
    parser.add_argument("--catalog-key-id", required=True)
    parser.add_argument("--pack-key-id", required=True)
    parser.add_argument("--installer-private-key", required=True, type=Path)
    parser.add_argument("--installer-key-id", required=True)
    parser.add_argument("--catalog-id", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument(
        "--asset-base-url",
        required=True,
        help=(
            "Direct HTTPS root or canonical immutable GitHub Release root "
            "containing release assets"
        ),
    )
    parser.add_argument("--cyxwiz-release", required=True)
    parser.add_argument("--bundle-version", required=True)
    parser.add_argument("--minimum-setup-version", required=True)
    parser.add_argument("--generated-utc", required=True)
    parser.add_argument("--expires-utc", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--openssl", default="openssl")
    return parser.parse_args(argv)


def _target_paths(
    values: Sequence[str], label: str, *, directories: bool
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        if not separator or key not in TARGET_BY_KEY or not raw_path:
            raise AlphaReleaseError(
                f"{label} must use a supported TARGET=PATH value"
            )
        if key in result:
            raise AlphaReleaseError(f"duplicate {label} target: {key}")
        input_path = Path(raw_path)
        if input_path.is_symlink():
            raise AlphaReleaseError(f"{label} for {key} must not be a link")
        path = input_path.resolve(strict=True)
        if (directories and not path.is_dir()) or (
            not directories and not path.is_file()
        ):
            expected = "directory" if directories else "regular file"
            raise AlphaReleaseError(f"{label} for {key} must be a {expected}")
        result[key] = path
    missing = set(TARGET_BY_KEY).difference(result)
    if missing:
        raise AlphaReleaseError(
            f"{label} is missing targets: {', '.join(sorted(missing))}"
        )
    return result


def _validate_release_identity(args: argparse.Namespace) -> None:
    if REPOSITORY.fullmatch(args.repository) is None:
        raise AlphaReleaseError("repository must be exact owner/name")
    try:
        args.asset_base_url = versioned_asset_base_url(
            args.asset_base_url, args.release_tag, args.repository
        )
    except ReleaseConfigurationError as error:
        raise AlphaReleaseError(str(error)) from error
    for label, value in (
        ("CyxWiz release", args.cyxwiz_release),
        ("bundle version", args.bundle_version),
        ("minimum setup version", args.minimum_setup_version),
    ):
        if VERSION.fullmatch(value) is None:
            raise AlphaReleaseError(f"{label} is invalid")


def _require_private_key(path: Path, label: str) -> Path:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise AlphaReleaseError(f"{label} must be a regular non-link file")
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AlphaReleaseError(
            f"cannot read release metadata {path}: {error}"
        ) from error
    if not isinstance(document, dict):
        raise AlphaReleaseError(f"release metadata is not an object: {path}")
    return document


def _validate_pack_matrix(repository_root: Path) -> None:
    manifests = repository_root / "bootstrap" / "catalogs" / "manifests"
    packs: list[Mapping[str, Any]] = []
    for path in sorted(manifests.glob("*.json")):
        document = _load_json(path)
        validate_pack_manifest(document)
        packs.append(document["signed"])
    for pack in packs:
        identity = (pack["platform"], pack["architecture"])
        if identity not in TARGET_BY_PACK:
            raise AlphaReleaseError(
                f"pack {pack['pack_id']} targets an unsupported alpha platform"
            )
    for identity, target in TARGET_BY_PACK.items():
        target_packs = [
            pack for pack in packs
            if (pack["platform"], pack["architecture"]) == identity
        ]
        bases = {
            pack["pack_id"] for pack in target_packs
            if pack["pack_kind"] == "base"
        }
        optional = [
            pack for pack in target_packs
            if pack["pack_kind"] == "backend_pack"
            and pack["companion_base_id"] in bases
        ]
        if not bases or not optional:
            raise AlphaReleaseError(
                f"{target.key} requires a base and matching optional pack"
            )


def _copy_asset(source: Path, assets: Path, observed: dict[str, str]) -> Path:
    name = source.name
    folded = name.casefold()
    previous = observed.get(folded)
    if previous is not None:
        raise AlphaReleaseError(f"release assets collide: {previous} and {name}")
    if source.is_symlink() or not source.is_file():
        raise AlphaReleaseError(f"release asset must be a regular file: {source}")
    observed[folded] = name
    destination = assets / name
    shutil.copyfile(source, destination)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installer_arguments(
    stage: Path,
    output: Path,
    target: ReleaseTarget,
    args: argparse.Namespace,
) -> argparse.Namespace:
    return installer_packager.parse_args(
        [
            str(stage), str(output),
            "--bundle-version", args.bundle_version,
            "--cyxwiz-release", args.cyxwiz_release,
            "--release-channel", "alpha",
            "--platform", target.platform,
            "--architecture", target.architecture,
            "--minimum-setup-version", args.minimum_setup_version,
            "--generated-utc", args.generated_utc,
            "--expires-utc", args.expires_utc,
        ]
    )


def _verify_installer_authority(
    descriptor: Path,
    trust_root: Path,
    key_id: str,
    openssl: str,
) -> None:
    pack_repository.verify_trusted_metadata_signature(
        _load_json(descriptor), trust_root, "installer", openssl, key_id
    )


def assemble(args: argparse.Namespace) -> Path:
    _validate_release_identity(args)
    stages = _target_paths(args.installer_stage, "installer stage", directories=True)
    setups = _target_paths(args.setup_package, "setup package", directories=False)
    for target in TARGETS:
        if setups[target.key].name != target.setup_name:
            raise AlphaReleaseError(
                f"setup package for {target.key} must be named {target.setup_name}"
            )
        installer_packager.staged_files(stages[target.key])
    output = args.output.resolve()
    if output.exists():
        raise AlphaReleaseError(f"output path already exists: {output}")
    for stage in stages.values():
        if output == stage or stage in output.parents:
            raise AlphaReleaseError("output must be outside every installer stage")
    catalog_key = _require_private_key(args.catalog_private_key, "catalog key")
    installer_key = _require_private_key(args.installer_private_key, "installer key")
    trust_root = args.trust_root.resolve(strict=True)
    base_url = args.asset_base_url
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=output.name + ".", dir=output.parent)
    )
    try:
        repository_root = temporary_root / "repository"
        repository_args = pack_repository.parse_args(
            [
                *(
                    item
                    for manifest in args.manifest
                    for item in ("--manifest", str(manifest))
                ),
                "--trust-root", str(trust_root),
                "--catalog-private-key", str(catalog_key),
                "--catalog-key-id", args.catalog_key_id,
                "--pack-key-id", args.pack_key_id,
                "--catalog-id", args.catalog_id,
                "--generated-utc", args.generated_utc,
                "--expires-utc", args.expires_utc,
                "--minimum-client-version", args.cyxwiz_release,
                "--base-url", base_url,
                "--hosted-layout", "flat",
                "--output", str(repository_root),
                "--openssl", args.openssl,
            ]
        )
        catalog_url = pack_repository.prepare_repository(repository_args)
        _validate_pack_matrix(repository_root)

        publication = temporary_root / "publication"
        assets = publication / "assets"
        assets.mkdir(parents=True)
        observed: dict[str, str] = {"sha256sums.txt": "SHA256SUMS.txt"}
        for asset in sorted((repository_root / "hosted").iterdir()):
            _copy_asset(asset, assets, observed)

        bundle_work = temporary_root / "bundles"
        for target in TARGETS:
            prepared_stage = temporary_root / "stages" / target.key
            shutil.copytree(stages[target.key], prepared_stage)
            runtime = prepared_stage / "runtime"
            if not runtime.is_dir() or runtime.is_symlink():
                raise AlphaReleaseError(
                    f"installer stage for {target.key} lacks a direct runtime directory"
                )
            shutil.rmtree(runtime)
            shutil.copytree(repository_root / "bootstrap", runtime)
            target_output = bundle_work / target.key
            archive, descriptor, _ = installer_packager.run(
                _installer_arguments(prepared_stage, target_output, target, args)
            )
            sign_descriptor(
                descriptor,
                installer_key,
                args.installer_key_id,
                args.openssl,
            )
            _verify_installer_authority(
                descriptor, trust_root, args.installer_key_id, args.openssl
            )
            _copy_asset(archive, assets, observed)
            _copy_asset(descriptor, assets, observed)
            _copy_asset(setups[target.key], assets, observed)

        payload_assets = sorted(assets.iterdir(), key=lambda path: path.name)
        checksums = "".join(
            f"{_sha256(path)}  {path.name}\n" for path in payload_assets
        )
        checksum_path = assets / "SHA256SUMS.txt"
        checksum_path.write_text(checksums, encoding="ascii", newline="\n")
        inventory = {
            "schema_version": 1,
            "kind": "cyxwiz-alpha-release-assets",
            "repository": args.repository,
            "release_tag": args.release_tag,
            "asset_base_url": base_url,
            "cyxwiz_release": args.cyxwiz_release,
            "bundle_version": args.bundle_version,
            "catalog_url": catalog_url,
            "assets": [
                {
                    "name": path.name,
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in payload_assets
            ],
        }
        (publication / "release-inventory.json").write_text(
            json.dumps(inventory, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        shutil.copytree(repository_root / "bootstrap", publication / "bootstrap")
        os.replace(publication, output)
    except Exception:
        if output.exists():
            raise AlphaReleaseError(
                "release output appeared during failed assembly; refusing replacement"
            )
        raise
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    try:
        output = assemble(parse_args(argv))
    except (
        AlphaReleaseError,
        OSError,
        ValueError,
        pack_repository.RepositoryError,
    ) as error:
        print(f"[ERROR] {error}")
        return 1
    print(f"[OK] Alpha release assets: {output / 'assets'}")
    print(f"[OK] Release inventory: {output / 'release-inventory.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
