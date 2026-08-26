from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_installer_package.py"
SPEC = importlib.util.spec_from_file_location("verify_installer_package", SCRIPT)
verify_installer_package = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_installer_package
SPEC.loader.exec_module(verify_installer_package)


class VerifyInstallerPackageTests(unittest.TestCase):
    def test_bootstrap_metadata_requires_matching_base_and_optional_manifests(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            stage = Path(temporary)
            trust = stage / "runtime" / "trust" / "trusted-keys.json"
            manifests = stage / "runtime" / "catalogs" / "manifests"
            trust.parent.mkdir(parents=True)
            manifests.mkdir(parents=True)
            trust.write_text(
                json.dumps({"schema_version": 1, "keys": []}),
                encoding="utf-8",
            )
            pack_documents = {
                "base-v1": "cpu",
                "opencl-v1": "opencl",
            }
            for pack_id, backend in pack_documents.items():
                (manifests / f"{pack_id}.json").write_text(
                    json.dumps({
                        "schema_version": 1,
                        "kind": "cyxwiz-backend-pack-manifest",
                        "signed": {"pack_id": pack_id, "backend": backend},
                    }),
                    encoding="utf-8",
                )
            (manifests.parent / "current.json").write_text(
                json.dumps({
                    "schema_version": 1,
                    "kind": "cyxwiz-backend-pack-catalog",
                    "signed": {
                        "packs": [
                            {"pack_id": pack_id} for pack_id in pack_documents
                        ]
                    },
                }),
                encoding="utf-8",
            )

            required = verify_installer_package.require_bootstrap_metadata(stage)

            self.assertEqual(4, len(required))

    def test_bootstrap_metadata_rejects_a_missing_catalog(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            with self.assertRaises(verify_installer_package.PackageSmokeError):
                verify_installer_package.require_bootstrap_metadata(Path(temporary))

    def test_inventory_is_relative_sized_and_hashed(self) -> None:
        stage = SCRIPT.parent
        files, installed_size, validation_size = (
            verify_installer_package.package_inventory(stage)
        )

        self.assertEqual(
            sum(path.stat().st_size for path in stage.rglob("*") if path.is_file()),
            installed_size,
        )
        self.assertEqual(0, validation_size)
        self.assertIn(SCRIPT.name, [item["path"] for item in files])
        self.assertTrue(all(item["role"] == "install_payload" for item in files))
        self.assertTrue(all(len(item["sha256"]) == 64 for item in files))

    def test_evidence_does_not_claim_accelerator_qualification(self) -> None:
        passed = {
            "name": "fixture",
            "arguments": [],
            "expected_exit_code": 0,
            "observed_exit_code": 0,
            "duration_ms": 1.0,
            "status": "passed",
        }
        evidence = verify_installer_package.build_evidence(
            "fixture-os",
            [{"path": "fixture", "role": "install_payload", "size_bytes": 7}],
            7,
            3,
            {"status": "passed", "duration_ms": 1.0},
            [passed] * 10,
        )

        self.assertEqual("fixture-os", evidence["artifact_id"])
        self.assertEqual("passed", evidence["result"])
        self.assertGreater(evidence["installed_size_bytes"], 0)
        self.assertEqual(3, evidence["validation_payload_size_bytes"])
        self.assertEqual(10, len(evidence["checks"]))
        self.assertEqual(
            {"cuda", "opencl", "oneapi"},
            {item["route"] for item in evidence["accelerator_routes"]},
        )
        self.assertTrue(
            all(item["status"] == "not_run" for item in evidence["accelerator_routes"])
        )


if __name__ == "__main__":
    unittest.main()
