from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import installer_alpha_publication_contract as contract
import verify_installer_alpha_publication as publication_verifier


class InstallerAlphaPublicationContractTests(unittest.TestCase):
    repository = "CYXWIZ-Lab/CYXWIZ"
    release_tag = "v0.2.0-alpha.1"
    cyxwiz_release = "0.2.0"
    bundle_version = "1.0.0"

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="cyxwiz-alpha-publication-contract-"
        )
        self.root = Path(self.temporary.name)
        self.upload = self.root / "upload"
        self.upload.mkdir()
        self.trust = self.root / "trusted-keys.json"
        self.trust.write_text("{}\n", encoding="utf-8")
        self.payload = {
            "catalog.json": b"catalog\n",
            "payload.bin": b"payload\n",
        }
        for name, content in self.payload.items():
            (self.upload / name).write_bytes(content)
        self.document = self._document()
        self._publish_controls()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _document(self) -> dict[str, object]:
        assets = [
            {
                "name": name,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for name, content in sorted(self.payload.items())
        ]
        body = {
            "kind": "cyxwiz-alpha-release-assets",
            "repository": self.repository,
            "release_tag": self.release_tag,
            "asset_base_url": (
                "https://packages.example.test/cyxwiz/" + self.release_tag
            ),
            "cyxwiz_release": self.cyxwiz_release,
            "bundle_version": self.bundle_version,
            "catalog_url": (
                "https://packages.example.test/cyxwiz/"
                + self.release_tag
                + "/catalog.json"
            ),
            "assets": assets,
        }
        return contract.build_signed_inventory(body, "installer-2026", "A" * 86)

    def _publish_controls(self) -> None:
        body = self.document["signed"]
        checksums = "".join(
            f"{entry['sha256']}  {entry['name']}\n"
            for entry in body["assets"]
        )
        (self.upload / contract.CHECKSUM_NAME).write_text(
            checksums, encoding="ascii", newline="\n"
        )
        (self.upload / contract.INVENTORY_NAME).write_text(
            json.dumps(self.document, indent=2) + "\n", encoding="utf-8"
        )

    def _validate(self, *, require_github: bool = False) -> dict[str, object]:
        with (
            mock.patch.object(
                contract, "verify_trusted_metadata_signature", return_value="key"
            ),
            mock.patch.object(
                contract,
                "_verify_repository_assets",
                return_value=set(self.payload),
            ),
            mock.patch.object(
                contract, "_verify_installer_assets", return_value=set()
            ),
        ):
            return contract.validate_upload_directory(
                self.upload,
                self.trust,
                self.repository,
                self.release_tag,
                self.cyxwiz_release,
                self.bundle_version,
                require_github=require_github,
            )

    def test_accepts_exact_signed_upload_directory(self) -> None:
        result = self._validate()
        self.assertEqual(self.release_tag, result["signed"]["release_tag"])

    def test_rejects_missing_and_extra_assets(self) -> None:
        (self.upload / "payload.bin").unlink()
        with self.assertRaisesRegex(contract.AlphaPublicationError, "asset set differs"):
            self._validate()
        (self.upload / "payload.bin").write_bytes(self.payload["payload.bin"])
        (self.upload / "unexpected.bin").write_bytes(b"unexpected")
        with self.assertRaisesRegex(contract.AlphaPublicationError, "asset set differs"):
            self._validate()

    def test_rejects_payload_and_checksum_drift(self) -> None:
        (self.upload / "payload.bin").write_bytes(b"modified")
        with self.assertRaisesRegex(contract.AlphaPublicationError, "differs from inventory"):
            self._validate()
        (self.upload / "payload.bin").write_bytes(self.payload["payload.bin"])
        (self.upload / contract.CHECKSUM_NAME).write_text(
            "0" * 64 + "  payload.bin\n", encoding="ascii"
        )
        with self.assertRaisesRegex(contract.AlphaPublicationError, "not canonical"):
            self._validate()

    def test_rejects_identity_change_and_non_github_publication(self) -> None:
        with self.assertRaisesRegex(contract.AlphaPublicationError, "release_tag"):
            contract.validate_upload_directory(
                self.upload,
                self.trust,
                self.repository,
                "v0.2.0-alpha.2",
                self.cyxwiz_release,
                self.bundle_version,
            )
        with self.assertRaisesRegex(contract.AlphaPublicationError, "canonical GitHub"):
            self._validate(require_github=True)

    def test_rejects_unknown_unsorted_colliding_and_reserved_assets(self) -> None:
        unknown = deepcopy(self.document)
        unknown["signed"]["unknown"] = True
        with self.assertRaisesRegex(contract.AlphaPublicationError, "unknown or missing"):
            contract.validate_inventory_document(unknown)

        unsorted = deepcopy(self.document)
        unsorted["signed"]["assets"].reverse()
        with self.assertRaisesRegex(contract.AlphaPublicationError, "not sorted"):
            contract.validate_inventory_document(unsorted)

        collision = deepcopy(self.document)
        duplicate = deepcopy(collision["signed"]["assets"][0])
        duplicate["name"] = duplicate["name"].upper()
        collision["signed"]["assets"].append(duplicate)
        collision["signed"]["assets"].sort(key=lambda item: item["name"])
        with self.assertRaisesRegex(contract.AlphaPublicationError, "case-insensitively"):
            contract.validate_inventory_document(collision)

        reserved = deepcopy(self.document)
        reserved["signed"]["assets"][0]["name"] = contract.CHECKSUM_NAME
        reserved["signed"]["assets"].sort(key=lambda item: item["name"])
        with self.assertRaisesRegex(contract.AlphaPublicationError, "reserved"):
            contract.validate_inventory_document(reserved)

    def test_github_draft_snapshot_requires_stable_exact_assets(self) -> None:
        assets = []
        for index, path in enumerate(sorted(self.upload.iterdir()), start=1):
            assets.append(
                {
                    "id": f"RA_fixture_{index}",
                    "name": path.name,
                    "size": path.stat().st_size,
                    "state": "uploaded",
                    "updatedAt": "2026-08-28T00:00:00Z",
                    "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        view = {
            "tagName": self.release_tag,
            "isDraft": True,
            "isPrerelease": True,
            "isImmutable": False,
            "assets": assets,
        }
        before = self.root / "before.json"
        after = self.root / "after.json"
        before.write_text(json.dumps(view), encoding="utf-8")
        after.write_text(json.dumps(view), encoding="utf-8")
        publication_verifier.validate_github_draft_release_views(
            before, after, self.upload, self.release_tag
        )

        changed = deepcopy(view)
        changed["assets"][0]["updatedAt"] = "2026-08-28T00:00:01Z"
        after.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(contract.AlphaPublicationError, "changed"):
            publication_verifier.validate_github_draft_release_views(
                before, after, self.upload, self.release_tag
            )

        published = deepcopy(view)
        published["isDraft"] = False
        before.write_text(json.dumps(published), encoding="utf-8")
        after.write_text(json.dumps(published), encoding="utf-8")
        with self.assertRaisesRegex(contract.AlphaPublicationError, "mutable draft"):
            publication_verifier.validate_github_draft_release_views(
                before, after, self.upload, self.release_tag
            )

        wrong_digest = deepcopy(view)
        wrong_digest["assets"][0]["digest"] = "sha256:" + "0" * 64
        before.write_text(json.dumps(wrong_digest), encoding="utf-8")
        after.write_text(json.dumps(wrong_digest), encoding="utf-8")
        with self.assertRaisesRegex(contract.AlphaPublicationError, "digest differs"):
            publication_verifier.validate_github_draft_release_views(
                before, after, self.upload, self.release_tag
            )


if __name__ == "__main__":
    unittest.main()
