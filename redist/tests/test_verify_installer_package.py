from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_installer_package.py"
SPEC = importlib.util.spec_from_file_location("verify_installer_package", SCRIPT)
verify_installer_package = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_installer_package
SPEC.loader.exec_module(verify_installer_package)


class VerifyInstallerPackageTests(unittest.TestCase):
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
        with mock.patch.object(
            verify_installer_package, "require_file", side_effect=lambda path: path
        ), mock.patch.object(
            verify_installer_package,
            "audit_dependencies",
            return_value={"status": "passed", "duration_ms": 1.0},
        ), mock.patch.object(
            verify_installer_package, "run_checked", return_value=passed
        ), mock.patch.object(
            verify_installer_package,
            "package_inventory",
            return_value=(
                [{"path": "fixture", "role": "install_payload", "size_bytes": 7}],
                7,
                3,
            ),
        ):
            evidence = verify_installer_package.verify(
                Path("fixture-stage"), "fixture-os"
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
