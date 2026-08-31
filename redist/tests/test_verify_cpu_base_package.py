from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
import stat
import zipfile
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_cpu_base_package.py"
SPEC = importlib.util.spec_from_file_location("verify_cpu_base_package", SCRIPT)
assert SPEC and SPEC.loader
verify_cpu_base_package = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_cpu_base_package
SPEC.loader.exec_module(verify_cpu_base_package)


class CpuBasePackageVerifierTests(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "Windows does not expose POSIX execute modes")
    def test_safe_extract_restores_native_executable_mode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cyxwiz-cpu-base-") as temporary:
            root = Path(temporary)
            archive = root / "native.zip"
            info = zipfile.ZipInfo("cyxwiz-runtime-bootstrapper")
            info.create_system = 3
            info.external_attr = 0o755 << 16
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr(info, b"native")

            destination = root / "extracted"
            verify_cpu_base_package.safe_extract(archive, destination)

            mode = (destination / info.filename).stat().st_mode
            self.assertTrue(mode & stat.S_IXUSR)

    def test_unsigned_ci_manifest_is_structurally_validated(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cyxwiz-cpu-base-") as temporary:
            root = Path(temporary)
            archive = root / "base.zip"
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr("payload.txt", "payload")
            signed = {
                "pack_id": "cyxwiz-base-0.2.0-1-win64-x86_64",
                "pack_kind": "base",
                "backend": "cpu",
                "package_version": "1",
                "platform": "win64",
                "architecture": "x86_64",
                "runtime_set_id": "arrayfire-3.10.0-win64-x86_64-v1",
                "cyxwiz_release": {"minimum": "0.2.0", "maximum": "0.2.0"},
                "arrayfire": {"version": "3.10.0", "abi": "arrayfire-3.10"},
                "companion_base_id": None,
                "conflicts": [],
                "compatibility": {
                    "device_kinds": ["cpu"],
                    "cpu_features": [],
                    "provider_types": ["arrayfire-cpu"],
                    "minimum_driver_versions": {},
                    "tested_driver_ranges": {},
                    "minimum_identity_confidence": "backend_local",
                    "recommendation_targets": [],
                    "operation_matrix_id": "cyxwiz-route-qualification-v1",
                    "training_scope": ["released-operation-matrix"],
                    "support_status": "diagnostic",
                },
                "components": [{
                    "path": "payload.txt",
                    "size": 7,
                    "sha256": verify_cpu_base_package.hashlib.sha256(
                        b"payload"
                    ).hexdigest(),
                    "source": "fixture",
                    "executable": False,
                }],
                "licenses": [{"component": "runtime", "path": "payload.txt"}],
                "archive": {
                    "file_name": archive.name,
                    "size": archive.stat().st_size,
                    "sha256": verify_cpu_base_package.sha256_file(archive),
                },
                "generated_utc": "2026-08-20T00:00:00Z",
            }
            manifest = {
                "schema_version": 1,
                "kind": "cyxwiz-backend-pack-manifest",
                "signed": signed,
                "signatures": [],
            }
            manifest_path = archive.with_suffix(".zip.manifest.json")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            archive.with_suffix(".zip.signed.json").write_bytes(
                verify_cpu_base_package.canonical_json_bytes(signed)
            )
            loaded = verify_cpu_base_package.load_manifest(manifest_path, archive)
            self.assertEqual([], loaded["signatures"])

    def test_engine_smoke_requires_explicit_cpu_and_isolation_evidence(self) -> None:
        verify_cpu_base_package.parse_engine_smoke(
            "package_smoke schema=1 status=pass effective_backend=cpu "
            "effective_device=0 runtime_isolation=pass checksum=70\n"
        )
        with self.assertRaisesRegex(
            verify_cpu_base_package.CpuBaseSmokeError, "did not report"
        ):
            verify_cpu_base_package.parse_engine_smoke(
                "package_smoke schema=1 status=pass effective_backend=cpu\n"
            )

    def test_probe_parser_rejects_cpu_fallback_masquerading_as_route(self) -> None:
        passing = (
            "probe_result schema=1 backend=cpu device_id=0 operation=constant "
            "status=pass effective_backend=1 effective_device=0\n"
        )
        verify_cpu_base_package.parse_probe(passing, "constant")
        with self.assertRaisesRegex(
            verify_cpu_base_package.CpuBaseSmokeError, "did not pass"
        ):
            verify_cpu_base_package.parse_probe(
                passing.replace("backend=cpu", "backend=opencl"), "constant"
            )

    def test_runtime_module_audit_requires_cpu_closure_inside_base(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cyxwiz-cpu-base-") as temporary:
            base = Path(temporary) / "base"
            runtime = base / "arrayfire" / "bin"
            runtime.mkdir(parents=True)
            modules = []
            for name in ("af.dll", "afcpu.dll", "mkl_rt.2.dll"):
                path = runtime / name
                path.write_bytes(b"fixture")
                modules.append(
                    f"runtime_module point=completed name={name} path='{path}'"
                )
            observed = verify_cpu_base_package.audit_runtime_modules(
                "\n".join(modules), base
            )
            self.assertEqual(3, len(observed))

    def test_runtime_module_audit_rejects_external_module(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cyxwiz-cpu-base-") as temporary:
            root = Path(temporary)
            base = root / "base"
            base.mkdir()
            external = root / "af.dll"
            external.write_bytes(b"fixture")
            output = (
                f"runtime_module point=completed name=af.dll path='{external}'"
            )
            with self.assertRaisesRegex(
                verify_cpu_base_package.CpuBaseSmokeError, "outside CPU base"
            ):
                verify_cpu_base_package.audit_runtime_modules(output, base)

    def test_macos_audit_rejects_any_non_system_absolute_dependency(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cyxwiz-cpu-base-") as temporary:
            base = Path(temporary) / "base"
            runtime = base / "arrayfire" / "lib"
            runtime.mkdir(parents=True)
            for name in ("libaf.dylib", "libafcpu.dylib"):
                (runtime / name).write_bytes(b"\xcf\xfa\xed\xfe" + b"fixture")
            output = (
                f"{runtime / 'libaf.dylib'}:\n"
                "\t/Applications/vendor/lib/libvendor.dylib "
                "(compatibility version 1.0.0, current version 1.0.0)\n"
            )
            completed = verify_cpu_base_package.subprocess.CompletedProcess(
                args=["otool"], returncode=0, stdout=output
            )

            with mock.patch.object(
                verify_cpu_base_package.subprocess, "run", return_value=completed
            ), self.assertRaisesRegex(
                verify_cpu_base_package.CpuBaseSmokeError,
                "non-system absolute dependency",
            ):
                verify_cpu_base_package.audit_macos_dependencies(base)


if __name__ == "__main__":
    unittest.main()
