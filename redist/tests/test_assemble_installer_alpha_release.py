from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


assembler = load_script("assemble_installer_alpha_release")


class AlphaReleaseAssemblerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.openssl = shutil.which("openssl")
        if not self.openssl:
            self.skipTest("OpenSSL is unavailable")
        self.temporary = tempfile.TemporaryDirectory(
            prefix="cyxwiz-alpha-release-test-"
        )
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.pack_key = self._key("pack")
        self.catalog_key = self._key("catalog")
        self.installer_key = self._key("installer")
        self.trust_root = self.inputs / "trusted-keys.json"
        self.trust_root.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "keys": [
                        self._trust_entry("pack-2026", self.pack_key, ["pack"]),
                        self._trust_entry(
                            "catalog-2026", self.catalog_key, ["catalog"]
                        ),
                        self._trust_entry(
                            "installer-2026", self.installer_key, ["installer"]
                        ),
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        self.stages: dict[str, Path] = {}
        self.setups: dict[str, Path] = {}
        self.manifests: list[Path] = []
        for target in assembler.TARGETS:
            self.stages[target.key] = self._stage(target)
            setup = self.inputs / "setups" / target.setup_name
            setup.parent.mkdir(parents=True, exist_ok=True)
            setup.write_bytes((target.key + "-setup").encode("ascii"))
            self.setups[target.key] = setup
            base_id = f"base-{target.key}"
            runtime_set = f"runtime-{target.key}"
            self.manifests.append(
                self._manifest(target, base_id, "cpu", None, runtime_set)
            )
            self.manifests.append(
                self._manifest(
                    target,
                    f"opencl-{target.key}",
                    "opencl",
                    base_id,
                    runtime_set,
                )
            )

    def tearDown(self) -> None:
        if hasattr(self, "temporary"):
            self.temporary.cleanup()

    def _key(self, name: str) -> Path:
        path = self.inputs / f"{name}.pem"
        subprocess.run(
            [self.openssl, "genpkey", "-algorithm", "ED25519", "-out", str(path)],
            check=True,
            capture_output=True,
        )
        return path

    def _raw_public_key(self, private_key: Path) -> bytes:
        result = subprocess.run(
            [
                self.openssl,
                "pkey",
                "-in", str(private_key),
                "-pubout",
                "-outform", "DER",
            ],
            check=True,
            capture_output=True,
        )
        prefix = assembler.pack_repository.ED25519_PUBLIC_KEY_DER_PREFIX
        self.assertTrue(result.stdout.startswith(prefix))
        return result.stdout[len(prefix) :]

    def _trust_entry(
        self, key_id: str, private_key: Path, roles: list[str]
    ) -> dict[str, object]:
        public_key = base64.urlsafe_b64encode(
            self._raw_public_key(private_key)
        ).decode("ascii").rstrip("=")
        return {
            "key_id": key_id,
            "algorithm": "ed25519",
            "public_key": public_key,
            "roles": roles,
            "revoked": False,
        }

    def _stage(self, target) -> Path:
        stage = self.inputs / "stages" / target.key
        suffix = ".exe" if target.platform == "windows" else ""
        files = {
            f"cyxwiz-installer{suffix}": b"installer",
            f"cyxwiz-backend-pack-installer{suffix}": b"helper",
            "resources/cyxwiz.png": b"logo",
            "runtime/trust/trusted-keys.json": b"fixture-trust\n",
            "runtime/catalogs/current.json": b"fixture-catalog\n",
        }
        for relative, content in files.items():
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            if path.name.startswith("cyxwiz-") and path.parent == stage:
                path.chmod(0o755)
        return stage

    def _manifest(
        self,
        target,
        pack_id: str,
        backend: str,
        companion_base_id: str | None,
        runtime_set_id: str,
        archive_name: str | None = None,
    ) -> Path:
        archive_name = archive_name or f"{pack_id}.zip"
        archive = self.inputs / "packs" / archive_name
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes((pack_id + "-archive").encode("ascii"))
        component_hash = hashlib.sha256(b"component").hexdigest()
        body = {
            "pack_id": pack_id,
            "pack_kind": "base" if backend == "cpu" else "backend_pack",
            "backend": backend,
            "package_version": "1",
            "platform": target.pack_platform,
            "architecture": target.architecture,
            "runtime_set_id": runtime_set_id,
            "cyxwiz_release": {"minimum": "0.2.0", "maximum": "0.2.0"},
            "arrayfire": {"version": "3.10.0", "abi": "arrayfire-3.10"},
            "companion_base_id": companion_base_id,
            "conflicts": [],
            "compatibility": {
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
            },
            "components": [
                {
                    "path": "runtime/component.bin",
                    "size": 9,
                    "sha256": component_hash,
                    "source": "arrayfire",
                    "executable": False,
                }
            ],
            "licenses": [
                {"component": "arrayfire", "path": "runtime/component.bin"}
            ],
            "archive": {
                "file_name": archive_name,
                "size": archive.stat().st_size,
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            },
            "generated_utc": "2026-08-28T00:00:00Z",
        }
        signature = assembler.pack_repository.sign_with_openssl(
            assembler.pack_repository.canonical_json_bytes(body),
            self.pack_key,
            self.openssl,
        )
        document = {
            "schema_version": 1,
            "kind": "cyxwiz-backend-pack-manifest",
            "signed": body,
            "signatures": [
                {
                    "key_id": "pack-2026",
                    "algorithm": "ed25519",
                    "value": base64.urlsafe_b64encode(signature)
                    .decode("ascii")
                    .rstrip("="),
                }
            ],
        }
        path = archive.with_suffix(".manifest.json")
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return path

    def arguments(self, manifests: list[Path] | None = None):
        values = [
            "--trust-root", str(self.trust_root),
            "--catalog-private-key", str(self.catalog_key),
            "--catalog-key-id", "catalog-2026",
            "--pack-key-id", "pack-2026",
            "--installer-private-key", str(self.installer_key),
            "--installer-key-id", "installer-2026",
            "--catalog-id", "alpha-2026-08",
            "--repository", "CYXWIZ-Lab/CYXWIZ",
            "--release-tag", "v0.2.0-alpha.1",
            "--asset-base-url",
            "https://packages.example.test/cyxwiz/v0.2.0-alpha.1",
            "--cyxwiz-release", "0.2.0",
            "--bundle-version", "1.0.0",
            "--minimum-setup-version", "1.0.0",
            "--generated-utc", "2026-08-28T00:00:00Z",
            "--expires-utc", "2026-11-28T00:00:00Z",
            "--output", str(self.root / "release"),
            "--openssl", self.openssl,
        ]
        for target in assembler.TARGETS:
            values.extend(
                ["--installer-stage", f"{target.key}={self.stages[target.key]}"]
            )
            values.extend(
                ["--setup-package", f"{target.key}={self.setups[target.key]}"]
            )
        for manifest in manifests or self.manifests:
            values.extend(["--manifest", str(manifest)])
        return assembler.parse_args(values)

    @unittest.skipIf(
        sys.platform == "win32",
        "Windows cannot represent POSIX executable bits for suffixless fixtures",
    )
    def test_assembles_complete_flat_signed_release_atomically(self) -> None:
        output = assembler.assemble(self.arguments())
        assets = output / "assets"
        inventory = json.loads(
            (output / "release-inventory.json").read_text(encoding="utf-8")
        )
        self.assertEqual("cyxwiz-alpha-release-assets", inventory["kind"])
        self.assertEqual(
            "https://packages.example.test/cyxwiz/v0.2.0-alpha.1",
            inventory["asset_base_url"],
        )
        self.assertEqual(
            "https://packages.example.test/cyxwiz/v0.2.0-alpha.1/"
            "cyxwiz-backend-catalog-alpha-2026-08.json",
            inventory["catalog_url"],
        )
        self.assertEqual(29, len(inventory["assets"]))
        self.assertEqual(30, len(list(assets.iterdir())))
        self.assertTrue((assets / "SHA256SUMS.txt").is_file())
        catalog_bytes = (output / "bootstrap/catalogs/current.json").read_bytes()
        self.assertIn(
            b"https://packages.example.test/cyxwiz/v0.2.0-alpha.1/",
            catalog_bytes,
        )
        self.assertNotIn(b"github.com", catalog_bytes)
        trust_bytes = (output / "bootstrap/trust/trusted-keys.json").read_bytes()
        self.assertEqual(self.trust_root.read_bytes(), trust_bytes)
        for target in assembler.TARGETS:
            descriptor_name = (
                f"cyxwiz-installer-0.2.0-1.0.0-"
                f"{target.platform}-{target.architecture}.descriptor.json"
            )
            descriptor = json.loads(
                (assets / descriptor_name).read_text(encoding="utf-8")
            )
            self.assertEqual("installer-2026", descriptor["signatures"][0]["key_id"])
            archive_name = descriptor_name.removesuffix(".descriptor.json") + ".zip"
            with zipfile.ZipFile(assets / archive_name) as archive:
                self.assertEqual(
                    catalog_bytes, archive.read("runtime/catalogs/current.json")
                )
                self.assertEqual(
                    trust_bytes, archive.read("runtime/trust/trusted-keys.json")
                )
        checksum_lines = (assets / "SHA256SUMS.txt").read_text(
            encoding="ascii"
        ).splitlines()
        self.assertEqual(29, len(checksum_lines))

    def test_rejects_incomplete_pack_matrix_without_output(self) -> None:
        filtered = [
            path for path in self.manifests
            if path.name != "opencl-macos-arm64.manifest.json"
        ]
        with self.assertRaisesRegex(
            assembler.AlphaReleaseError,
            "macos-arm64 requires a base and matching optional pack",
        ):
            assembler.assemble(self.arguments(filtered))
        self.assertFalse((self.root / "release").exists())

    @unittest.skipIf(
        sys.platform == "win32",
        "Windows cannot represent POSIX executable bits for suffixless fixtures",
    )
    def test_rejects_installer_key_outside_trust_root(self) -> None:
        other_key = self._key("other-installer")
        arguments = self.arguments()
        arguments.installer_private_key = other_key
        with self.assertRaisesRegex(
            assembler.pack_repository.RepositoryError,
            "no valid trusted",
        ):
            assembler.assemble(arguments)
        self.assertFalse((self.root / "release").exists())

    def test_rejects_checksum_asset_collision_without_output(self) -> None:
        target = assembler.TARGET_BY_KEY["windows-x64"]
        colliding = self._manifest(
            target,
            "base-windows-x64",
            "cpu",
            None,
            "runtime-windows-x64",
            "SHA256SUMS.txt",
        )
        manifests = [
            path for path in self.manifests
            if path.name != "base-windows-x64.manifest.json"
        ]
        manifests.append(colliding)
        with self.assertRaisesRegex(
            assembler.AlphaReleaseError,
            "release assets collide",
        ):
            assembler.assemble(self.arguments(manifests))
        self.assertFalse((self.root / "release").exists())

    def test_rejects_non_direct_asset_url_without_output(self) -> None:
        arguments = self.arguments()
        arguments.asset_base_url = "http://packages.example.test/cyxwiz/alpha"
        with self.assertRaisesRegex(
            assembler.AlphaReleaseError,
            "use HTTPS",
        ):
            assembler.assemble(arguments)
        self.assertFalse((self.root / "release").exists())


if __name__ == "__main__":
    unittest.main()
