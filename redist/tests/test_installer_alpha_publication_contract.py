from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
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
import prepare_backend_pack_repository as pack_repository
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

    def test_cpu_only_inventory_kind_is_explicit(self) -> None:
        document = deepcopy(self.document)
        document["signed"]["kind"] = "cyxwiz-alpha-cpu-release-assets"
        contract.validate_inventory_document(document)
        document["signed"]["kind"] = "cyxwiz-alpha-any-release-assets"
        with self.assertRaisesRegex(contract.AlphaPublicationError, "kind is invalid"):
            contract.validate_inventory_document(document)

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


class PluginPackPublicationTests(unittest.TestCase):
    """A release upload carrying a plugin pack (schema-2 manifest).

    The pack manifests, catalog and inventory are signed with throwaway
    Ed25519 keys and verified for real; only the installer bundles are left
    to their own tests.
    """

    repository = "CYXWIZ-Lab/CYXWIZ"
    release_tag = "v0.2.0-alpha.1"
    cyxwiz_release = "0.2.0"
    bundle_version = "1.0.0"
    base_url = "https://packages.example.test/cyxwiz/v0.2.0-alpha.1"
    catalog_id = "alpha-2026-11"
    plugin_id = "cyxwiz-plugin-mujoco-1.0.0-1-win64-x86_64"
    bases = {
        ("win64", "x86_64"): "cyxwiz-base-0.2.0-1-win64-x86_64",
        ("linux64", "x86_64"): "cyxwiz-base-0.2.0-1-linux64-x86_64",
        ("macos", "x86_64"): "cyxwiz-base-0.2.0-1-macos-x86_64",
        ("macos", "arm64"): "cyxwiz-base-0.2.0-1-macos-arm64",
    }

    def setUp(self) -> None:
        self.openssl = shutil.which("openssl")
        if self.openssl is None:
            self.skipTest("OpenSSL is not available")
        self.temporary = tempfile.TemporaryDirectory(
            prefix="cyxwiz-alpha-plugin-publication-"
        )
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.keys: dict[str, Path] = {}
        for role in ("pack", "catalog", "installer"):
            path = self.root / f"{role}.pem"
            subprocess.run(
                [self.openssl, "genpkey", "-algorithm", "ED25519", "-out", str(path)],
                check=True,
                capture_output=True,
            )
            self.keys[role] = path
        self.trust = self.root / "trusted-keys.json"
        self.trust.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "keys": [
                        {
                            "key_id": f"{role}-2026",
                            "algorithm": "ed25519",
                            "public_key": self._public_key(path),
                            "roles": [role],
                            "revoked": False,
                        }
                        for role, path in self.keys.items()
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        if hasattr(self, "temporary"):
            self.temporary.cleanup()

    def _public_key(self, private_key: Path) -> str:
        result = subprocess.run(
            [self.openssl, "pkey", "-in", str(private_key), "-pubout", "-outform", "DER"],
            check=True,
            capture_output=True,
        )
        prefix = pack_repository.ED25519_PUBLIC_KEY_DER_PREFIX
        self.assertTrue(result.stdout.startswith(prefix))
        raw = result.stdout[len(prefix):]
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    def _signature(self, body: dict[str, object], role: str) -> dict[str, str]:
        value = pack_repository.sign_with_openssl(
            pack_repository.canonical_json_bytes(body), self.keys[role], self.openssl
        )
        return {
            "key_id": f"{role}-2026",
            "algorithm": "ed25519",
            "value": base64.urlsafe_b64encode(value).decode("ascii").rstrip("="),
        }

    def _manifest(
        self,
        pack_id: str,
        identity: tuple[str, str],
        kind: str,
        companion: str | None = None,
        backend: str = "cpu",
    ) -> Path:
        archive = self.inputs / f"{pack_id}.zip"
        archive.write_bytes((pack_id + "-archive").encode("ascii"))
        component = hashlib.sha256(b"component").hexdigest()
        body: dict[str, object] = {
            "pack_id": pack_id,
            "pack_kind": kind,
            "package_version": "1",
            "platform": identity[0],
            "architecture": identity[1],
            "cyxwiz_release": {"minimum": "0.2.0", "maximum": "0.2.0"},
            "companion_base_id": companion,
            "conflicts": [],
            "archive": {
                "file_name": archive.name,
                "size": archive.stat().st_size,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            },
            "generated_utc": "2026-11-24T00:00:00Z",
        }
        if kind == "plugin_pack":
            # Schema 2: the plugin and its own runtime, no ArrayFire or device fields
            schema_version = 2
            body["plugin_id"] = "com.cyxwiz.simulation.mujoco"
            body["support_status"] = "supported"
            body["components"] = [
                {"path": "plugin.json", "size": 9, "sha256": component,
                 "source": "cyxwiz-plugin", "executable": False},
                {"path": "bin/mujoco_plugin.dll", "size": 9, "sha256": component,
                 "source": "cyxwiz-plugin", "executable": True},
                {"path": "bin/mujoco.dll", "size": 9, "sha256": component,
                 "source": "plugin-runtime", "executable": True},
                {"path": "THIRD_PARTY_LICENSES/MuJoCo/MUJOCO_LICENSE.txt", "size": 9,
                 "sha256": component, "source": "mujoco", "executable": False},
            ]
            body["licenses"] = [
                {"component": "mujoco", "path": "THIRD_PARTY_LICENSES/MuJoCo/MUJOCO_LICENSE.txt"}
            ]
        else:
            schema_version = 1
            body["backend"] = backend
            body["runtime_set_id"] = f"arrayfire-3.10.0-{identity[0]}-{identity[1]}-v1"
            body["arrayfire"] = {"version": "3.10.0", "abi": "arrayfire-3.10"}
            body["compatibility"] = {
                "device_kinds": ["cpu" if backend == "cpu" else "gpu"],
                "cpu_features": [],
                "provider_types": [],
                "minimum_driver_versions": {},
                "tested_driver_ranges": {},
                "minimum_identity_confidence": "backend_local",
                "recommendation_targets": [backend],
                "operation_matrix_id": "released-operations-v1",
                "training_scope": ["released-operation-matrix"],
                "support_status": "supported",
            }
            body["components"] = [
                {"path": "runtime/component.bin", "size": 9, "sha256": component,
                 "source": "arrayfire", "executable": False},
            ]
            body["licenses"] = [{"component": "arrayfire", "path": "runtime/component.bin"}]
        document = {
            "schema_version": schema_version,
            "kind": "cyxwiz-backend-pack-manifest",
            "signed": body,
            "signatures": [self._signature(body, "pack")],
        }
        path = self.inputs / f"{pack_id}.zip.manifest.json"
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return path

    def _manifests(self, *, accelerator: bool = False) -> list[Path]:
        manifests = [
            self._manifest(base_id, identity, "base")
            for identity, base_id in self.bases.items()
        ]
        windows = ("win64", "x86_64")
        manifests.append(
            self._manifest(self.plugin_id, windows, "plugin_pack", self.bases[windows])
        )
        if accelerator:
            manifests.append(
                self._manifest(
                    "cyxwiz-af-opencl-0.2.0-1-win64-x86_64", windows, "backend_pack",
                    self.bases[windows], backend="opencl",
                )
            )
        return manifests

    def _upload(self, manifests: list[Path], *, cpu_only: bool) -> Path:
        output = self.root / "repository"
        values = [
            "--trust-root", str(self.trust),
            "--catalog-private-key", str(self.keys["catalog"]),
            "--catalog-key-id", "catalog-2026",
            "--pack-key-id", "pack-2026",
            "--catalog-id", self.catalog_id,
            "--generated-utc", "2026-11-24T00:00:00Z",
            "--expires-utc", "2027-02-24T00:00:00Z",
            "--minimum-client-version", self.cyxwiz_release,
            "--base-url", self.base_url,
            "--hosted-layout", "flat",
            "--output", str(output),
            "--openssl", self.openssl,
        ]
        for manifest in manifests:
            values.extend(["--manifest", str(manifest)])
        self.catalog_url = pack_repository.prepare_repository(
            pack_repository.parse_args(values)
        )
        upload = self.root / "upload"
        shutil.copytree(output / "hosted", upload)
        self.cpu_only = cpu_only
        self._sign_inventory(upload)
        return upload

    def _sign_inventory(self, upload: Path) -> None:
        reserved = {contract.INVENTORY_NAME, contract.CHECKSUM_NAME}
        assets = sorted(
            (path for path in upload.iterdir() if path.name not in reserved),
            key=lambda path: path.name,
        )
        digests = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in assets}
        body = {
            "kind": (
                "cyxwiz-alpha-cpu-release-assets" if self.cpu_only
                else "cyxwiz-alpha-release-assets"
            ),
            "repository": self.repository,
            "release_tag": self.release_tag,
            "asset_base_url": self.base_url,
            "cyxwiz_release": self.cyxwiz_release,
            "bundle_version": self.bundle_version,
            "catalog_url": self.catalog_url,
            "assets": [
                {"name": path.name, "size": path.stat().st_size, "sha256": digests[path.name]}
                for path in assets
            ],
        }
        signature = self._signature(body, "installer")
        document = contract.build_signed_inventory(
            body, signature["key_id"], signature["value"]
        )
        (upload / contract.CHECKSUM_NAME).write_text(
            "".join(f"{digests[path.name]}  {path.name}\n" for path in assets),
            encoding="ascii",
            newline="\n",
        )
        (upload / contract.INVENTORY_NAME).write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )

    def _validate(self, upload: Path) -> dict[str, object]:
        # Installer bundles have their own tests; this upload carries none.
        with mock.patch.object(contract, "_verify_installer_assets", return_value=set()):
            return contract.validate_upload_directory(
                upload,
                self.trust,
                self.repository,
                self.release_tag,
                self.cyxwiz_release,
                self.bundle_version,
                openssl=self.openssl,
            )

    def test_cpu_only_release_carries_a_plugin_pack(self) -> None:
        # A plugin pack is not an optional compute pack: a CPU-only release may carry it.
        upload = self._upload(self._manifests(), cpu_only=True)
        document = self._validate(upload)
        names = {entry["name"] for entry in document["signed"]["assets"]}
        self.assertEqual("cyxwiz-alpha-cpu-release-assets", document["signed"]["kind"])
        self.assertIn(f"{self.plugin_id}.json", names)
        self.assertIn(f"{self.plugin_id}.zip", names)

    def test_plugin_pack_does_not_make_a_full_release(self) -> None:
        upload = self._upload(self._manifests(), cpu_only=False)
        with self.assertRaisesRegex(contract.AlphaPublicationError, "no optional packs"):
            self._validate(upload)

    def test_full_release_carries_a_plugin_pack_beside_an_accelerator_pack(self) -> None:
        upload = self._upload(self._manifests(accelerator=True), cpu_only=False)
        document = self._validate(upload)
        self.assertEqual("cyxwiz-alpha-release-assets", document["signed"]["kind"])
        # The CPU-only kind still refuses the accelerator pack, never the plugin pack
        self.cpu_only = True
        self._sign_inventory(upload)
        with self.assertRaisesRegex(contract.AlphaPublicationError, "must not contain optional"):
            self._validate(upload)

    def test_rejects_a_plugin_archive_that_differs_from_its_manifest(self) -> None:
        upload = self._upload(self._manifests(), cpu_only=True)
        (upload / f"{self.plugin_id}.zip").write_bytes(b"replaced plugin archive")
        # Signed into the inventory, so only the pack's own manifest can catch it
        self._sign_inventory(upload)
        with self.assertRaisesRegex(
            contract.AlphaPublicationError, "pack archive differs from its signed manifest"
        ):
            self._validate(upload)

    def test_rejects_a_plugin_pack_whose_base_is_not_in_the_release(self) -> None:
        upload = self._upload(self._manifests(), cpu_only=True)
        # A correctly signed plugin manifest and catalog naming another base
        manifest_path = upload / f"{self.plugin_id}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["signed"]["companion_base_id"] = "cyxwiz-base-0.2.0-9-win64-x86_64"
        manifest["signatures"] = [self._signature(manifest["signed"], "pack")]
        manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)
        catalog_path = upload / f"cyxwiz-backend-catalog-{self.catalog_id}.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        for entry in catalog["signed"]["packs"]:
            if entry["pack_id"] == self.plugin_id:
                entry["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
        catalog["signatures"] = [self._signature(catalog["signed"], "catalog")]
        catalog_path.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
        self._sign_inventory(upload)
        with self.assertRaisesRegex(
            contract.AlphaPublicationError, f"plugin pack {self.plugin_id} requires a base absent"
        ):
            self._validate(upload)


if __name__ == "__main__":
    unittest.main()
