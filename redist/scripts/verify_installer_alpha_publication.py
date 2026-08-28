#!/usr/bin/env python3
"""Verify a signed installer alpha and one stable GitHub draft snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from installer_alpha_publication_contract import (
    MAX_ASSETS,
    MAX_INVENTORY_BYTES,
    AlphaPublicationError,
    validate_upload_directory,
)


def _load_release_view(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise AlphaPublicationError("GitHub draft metadata must be a regular file")
    if path.stat().st_size > 4 * MAX_INVENTORY_BYTES:
        raise AlphaPublicationError("GitHub draft metadata exceeds its byte bound")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AlphaPublicationError("GitHub draft metadata is invalid JSON") from error
    if not isinstance(document, dict):
        raise AlphaPublicationError("GitHub draft metadata must be an object")
    return document


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _draft_asset_snapshot(
    document: Any, release_tag: str, expected_files: Mapping[str, Path]
) -> tuple[tuple[str, str, int, str, str | None], ...]:
    expected_keys = {"tagName", "isDraft", "isPrerelease", "isImmutable", "assets"}
    if not isinstance(document, dict) or set(document) != expected_keys:
        raise AlphaPublicationError(
            "GitHub draft release contains unknown or missing fields"
        )
    if (
        document["tagName"] != release_tag
        or document["isDraft"] is not True
        or document["isPrerelease"] is not True
        or document["isImmutable"] is not False
    ):
        raise AlphaPublicationError(
            "GitHub release must be the exact mutable draft prerelease"
        )
    assets = document["assets"]
    if not isinstance(assets, list) or not 1 <= len(assets) <= MAX_ASSETS + 2:
        raise AlphaPublicationError("GitHub draft asset count is invalid")

    snapshot: list[tuple[str, str, int, str, str | None]] = []
    names: set[str] = set()
    for raw in assets:
        if not isinstance(raw, dict):
            raise AlphaPublicationError("GitHub draft asset metadata is invalid")
        asset_id = raw.get("id")
        name = raw.get("name")
        size = raw.get("size")
        state = raw.get("state")
        updated = raw.get("updatedAt")
        digest = raw.get("digest")
        if (
            not isinstance(asset_id, str)
            or not asset_id
            or len(asset_id) > 128
            or not isinstance(name, str)
            or name not in expected_files
            or name.casefold() in names
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size != expected_files[name].stat().st_size
            or state != "uploaded"
            or not isinstance(updated, str)
            or not updated
            or len(updated) > 64
            or (digest is not None and not isinstance(digest, str))
        ):
            raise AlphaPublicationError("GitHub draft asset metadata differs")
        names.add(name.casefold())
        local_sha256 = _sha256(expected_files[name])
        if digest not in {None, "", f"sha256:{local_sha256}"}:
            raise AlphaPublicationError(f"GitHub draft digest differs for {name}")
        snapshot.append((asset_id, name, size, updated, digest or None))
    if {item[1] for item in snapshot} != set(expected_files):
        raise AlphaPublicationError("GitHub draft release has missing or extra assets")
    return tuple(sorted(snapshot, key=lambda item: item[1]))


def validate_github_draft_release_views(
    before_path: Path,
    after_path: Path,
    upload_directory: Path,
    release_tag: str,
) -> None:
    root = upload_directory.resolve(strict=True)
    expected_files = {
        path.name: path
        for path in root.iterdir()
        if path.is_file() and not path.is_symlink()
    }
    before = _draft_asset_snapshot(
        _load_release_view(before_path), release_tag, expected_files
    )
    after = _draft_asset_snapshot(
        _load_release_view(after_path), release_tag, expected_files
    )
    if before != after:
        raise AlphaPublicationError(
            "GitHub draft assets changed while publication validation ran"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--trust-root", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--cyxwiz-release", required=True)
    parser.add_argument("--bundle-version", required=True)
    parser.add_argument("--require-github", action="store_true")
    parser.add_argument("--release-view-before", type=Path)
    parser.add_argument("--release-view-after", type=Path)
    parser.add_argument("--openssl", default="openssl")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if (args.release_view_before is None) != (args.release_view_after is None):
            raise AlphaPublicationError(
                "both GitHub draft metadata snapshots are required together"
            )
        validate_upload_directory(
            args.directory,
            args.trust_root,
            args.repository,
            args.release_tag,
            args.cyxwiz_release,
            args.bundle_version,
            openssl=args.openssl,
            require_github=args.require_github,
        )
        if args.release_view_before is not None:
            validate_github_draft_release_views(
                args.release_view_before,
                args.release_view_after,
                args.directory,
                args.release_tag,
            )
    except (OSError, ValueError) as error:
        print(f"[ERROR] {error}")
        return 1
    print(f"[OK] Verified installer alpha upload: {args.directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
