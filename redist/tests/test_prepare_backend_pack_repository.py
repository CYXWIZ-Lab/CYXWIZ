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
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prepare_backend_pack_repository.py"
)
SPEC = importlib.util.spec_from_file_location(
    "prepare_backend_pack_repository", SCRIPT
)
assert SPEC and SPEC.loader
repository = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = repository
SPEC.loader.exec_module(repository)


class BackendPackRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.openssl = shutil.which("openssl")
        if self.openssl is None:
            self.skipTest("OpenSSL is not available")
        self.temporary = tempfile.TemporaryDirectory(
            prefix="cyxwiz-repository-test-"
        )
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.catalog_key = self.root / "catalog.pem"
        self.pack_key = self.root / "pack.pem"
        self.other_key = self.root / "other.pem"
        for path in (self.catalog_key, self.pack_key, self.other_key):
            subprocess.run(
                [
                    self.openssl,
                    "genpkey",
                    "-algorithm",
                    "ED25519",
                    "-out",
                    str(path),
                ],
                check=True,
                capture_output=True,
            )
        self.trust_root = self.root / "trusted-keys.json"
        self.trust_root.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "keys": [
                        self._trust_entry(
                            "catalog-2026", self.catalog_key, ["catalog"]
                        ),
                        self._trust_entry(
                            "pack-2026", self.pack_key, ["pack"]
                        ),
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

    def _raw_public_key(self, private_key: Path) -> bytes:
        result = subprocess.run(
            [
                self.openssl,
                "pkey",
                "-in",
                str(private_key),
                "-pubout",
                "-outform",
                "DER",
            ],
            check=True,
            capture_output=True,
        )
        self.assertTrue(
            result.stdout.startswith(repository.ED25519_PUBLIC_KEY_DER_PREFIX)
        )
        return result.stdout[len(repository.ED25519_PUBLIC_KEY_DER_PREFIX) :]

    def _trust_entry(
        self, key_id: str, private_key: Path, roles: list[str]
    ) -> dict[str, object]:
        encoded = base64.urlsafe_b64encode(
            self._raw_public_key(private_key)
        ).decode("ascii").rstrip("=")
        return {
            "key_id": key_id,
            "algorithm": "ed25519",
            "public_key": encoded,
            "roles": roles,
            "revoked": False,
        }

    def _manifest(
        self,
        pack_id: str,
        backend: str,
        companion_base_id: str | None,
        *,
        private_key: Path | None = None,
        signing_key_id: str = "pack-2026",
        runtime_set_id: str = "runtime-310",
    ) -> Path:
        archive_name = f"{pack_id}.zip"
        archive = self.inputs / archive_name
        archive.write_bytes((pack_id + "-archive").encode("ascii"))
        archive_bytes = archive.read_bytes()
        component_hash = hashlib.sha256(b"component").hexdigest()
        body = {
            "pack_id": pack_id,
            "pack_kind": "base" if backend == "cpu" else "backend_pack",
            "backend": backend,
            "package_version": "1",
            "platform": "win64",
            "architecture": "x86_64",
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
                {
                    "component": "arrayfire",
                    "path": "runtime/component.bin",
                }
            ],
            "archive": {
                "file_name": archive_name,
                "size": len(archive_bytes),
                "sha256": hashlib.sha256(archive_bytes).hexdigest(),
            },
            "generated_utc": "2026-08-25T10:00:00Z",
        }
        signature = repository.sign_with_openssl(
            repository.canonical_json_bytes(body),
            private_key or self.pack_key,
            self.openssl,
        )
        document = {
            "schema_version": 1,
            "kind": "cyxwiz-backend-pack-manifest",
            "signed": body,
            "signatures": [
                {
                    "key_id": signing_key_id,
                    "algorithm": "ed25519",
                    "value": base64.urlsafe_b64encode(signature)
                    .decode("ascii")
                    .rstrip("="),
                }
            ],
        }
        path = self.inputs / f"{pack_id}.manifest.json"
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return path

    def _args(self, manifests: list[Path], output: Path | None = None):
        values = [
            "--trust-root",
            str(self.trust_root),
            "--catalog-private-key",
            str(self.catalog_key),
            "--catalog-key-id",
            "catalog-2026",
            "--pack-key-id",
            "pack-2026",
            "--catalog-id",
            "alpha-2026-08",
            "--generated-utc",
            "2026-08-25T12:00:00Z",
            "--expires-utc",
            "2026-09-25T12:00:00Z",
            "--minimum-client-version",
            "0.2.0",
            "--base-url",
            "https://packages.example.test/cyxwiz/alpha",
            "--output",
            str(output or (self.root / "repository")),
            "--openssl",
            self.openssl,
        ]
        for manifest in manifests:
            values.extend(["--manifest", str(manifest)])
        return repository.parse_args(values)

    def test_assembles_matching_hosted_and_bootstrap_metadata(self) -> None:
        base = self._manifest("base-v1", "cpu", None)
        opencl = self._manifest("opencl-v1", "opencl", "base-v1")
        output = self.root / "repository"

        catalog_url = repository.prepare_repository(
            self._args([opencl, base], output)
        )

        self.assertEqual(
            "https://packages.example.test/cyxwiz/alpha/catalogs/current.json",
            catalog_url,
        )
        hosted = output / "hosted"
        bootstrap = output / "bootstrap"
        catalog_bytes = (hosted / "catalogs" / "current.json").read_bytes()
        self.assertEqual(
            catalog_bytes,
            (bootstrap / "catalogs" / "current.json").read_bytes(),
        )
        catalog = json.loads(catalog_bytes)
        repository.validate_catalog(catalog)
        self.assertEqual(
            ["base-v1", "opencl-v1"],
            [entry["pack_id"] for entry in catalog["signed"]["packs"]],
        )
        self.assertTrue((hosted / "catalogs" / "manifests" / "base-v1.zip").is_file())
        self.assertTrue((hosted / "catalogs" / "manifests" / "opencl-v1.zip").is_file())
        self.assertFalse((hosted / "trust").exists())
        self.assertTrue((bootstrap / "trust" / "trusted-keys.json").is_file())
        self.assertTrue((bootstrap / "catalogs" / "manifests" / "base-v1.json").is_file())
        self.assertFalse((bootstrap / "catalogs" / "manifests" / "base-v1.zip").exists())

    def test_rejects_archive_tampering_without_publishing_output(self) -> None:
        base = self._manifest("base-v1", "cpu", None)
        (self.inputs / "base-v1.zip").write_bytes(b"tampered")
        output = self.root / "repository"

        with self.assertRaisesRegex(
            repository.RepositoryError, "Archive (size|SHA-256) differs"
        ):
            repository.prepare_repository(self._args([base], output))

        self.assertFalse(output.exists())

    def test_rejects_optional_pack_without_its_exact_base(self) -> None:
        opencl = self._manifest("opencl-v1", "opencl", "base-v1")
        output = self.root / "repository"

        with self.assertRaisesRegex(
            repository.RepositoryError, "requires absent base"
        ):
            repository.prepare_repository(self._args([opencl], output))

        self.assertFalse(output.exists())

    def test_rejects_untrusted_pack_signature(self) -> None:
        base = self._manifest(
            "base-v1",
            "cpu",
            None,
            private_key=self.other_key,
        )

        with self.assertRaisesRegex(
            repository.RepositoryError, "no valid trusted"
        ):
            repository.prepare_repository(self._args([base]))

    def test_rejects_catalog_key_that_does_not_match_trust_root(self) -> None:
        base = self._manifest("base-v1", "cpu", None)
        args = self._args([base])
        args.catalog_private_key = self.other_key

        with self.assertRaisesRegex(
            repository.RepositoryError, "no valid trusted"
        ):
            repository.prepare_repository(args)

    def test_rejects_non_direct_repository_url(self) -> None:
        base = self._manifest("base-v1", "cpu", None)
        args = self._args([base])
        args.base_url = "https://user@example.test/repository?token=secret"

        with self.assertRaisesRegex(
            repository.RepositoryError, "direct HTTPS"
        ):
            repository.prepare_repository(args)


if __name__ == "__main__":
    unittest.main()
