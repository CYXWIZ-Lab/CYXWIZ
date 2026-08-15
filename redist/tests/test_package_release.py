from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "package_release.py"
SPEC = importlib.util.spec_from_file_location("package_release", SCRIPT)
assert SPEC and SPEC.loader
package_release = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = package_release
SPEC.loader.exec_module(package_release)

SIGN_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sign_pack_manifest.py"
SIGN_SPEC = importlib.util.spec_from_file_location("sign_pack_manifest", SIGN_SCRIPT)
assert SIGN_SPEC and SIGN_SPEC.loader
sign_pack_manifest = importlib.util.module_from_spec(SIGN_SPEC)
sys.modules[SIGN_SPEC.name] = sign_pack_manifest
SIGN_SPEC.loader.exec_module(sign_pack_manifest)


class PackageReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="cyxwiz-package-test-")
        self.root = Path(self.temporary.name)
        self.script = self.root / "redist" / "scripts" / "package_release.py"
        self.script.parent.mkdir(parents=True)
        templates = self.root / "redist" / "templates"
        templates.mkdir()
        template = "version={{VERSION}} platform={{PLATFORM}} backends={{BACKENDS}} af={{ARRAYFIRE_VERSION}}\n"
        (templates / "README_MINIMAL.md").write_text(template, encoding="utf-8")
        (templates / "README_FULL.md").write_text(template, encoding="utf-8")
        (templates / "README_BASE.md").write_text(template, encoding="utf-8")

        build = self.root / "build" / "bin" / "Release"
        build.mkdir(parents=True)
        for name in (
            "cyxwiz-engine.exe",
            "cyxwiz-route-probe.exe",
            "cyxwiz-installer.exe",
            "cyxwiz-runtime-bootstrapper.exe",
            "cyxwiz-backend-pack-installer.exe",
            "cyxwiz-backend.dll",
            "fmt.dll",
            "python312.dll",
        ):
            (build / name).write_bytes(name.encode("ascii"))
        resources = self.root / "cyxwiz-engine" / "resources"
        resources.mkdir(parents=True)
        (resources / "resource.txt").write_text("resource", encoding="ascii")
        (self.root / "LICENSE").write_text("license", encoding="ascii")
        notices = self.root / "build" / "vcpkg_installed" / "x64-windows" / "share" / "fmt"
        notices.mkdir(parents=True)
        (notices / "copyright").write_text("fmt license", encoding="ascii")

        include = self.root / "cyxwiz-backend" / "include" / "cyxwiz"
        include.mkdir(parents=True)
        (include / "version.h").write_text(
            "#define CYXWIZ_VERSION_MAJOR 1\n"
            "#define CYXWIZ_VERSION_MINOR 2\n"
            "#define CYXWIZ_VERSION_PATCH 3\n",
            encoding="ascii",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def args(self, *values: str):
        return package_release.parse_args([*values, "--stage-only"])

    def artifact_args(self, *values: str):
        return package_release.parse_args([*values])

    def create_arrayfire(self, oneapi: bool = False, opencl: bool = False) -> Path:
        root = self.root / "arrayfire"
        library = root / "lib"
        licenses = root / "LICENSES"
        include = root / "include" / "af"
        library.mkdir(parents=True)
        licenses.mkdir()
        include.mkdir(parents=True)
        (licenses / "BSD.txt").write_text("license", encoding="ascii")
        (include / "version.h").write_text('#define AF_VERSION "3.10.0"\n', encoding="ascii")
        for name in ("af.dll", "afcpu.dll", "mkl_rt.2.dll"):
            (library / name).write_bytes(name.encode("ascii"))
        if opencl:
            (library / "afopencl.dll").write_bytes(b"opencl")
        if oneapi:
            for name in (
                "afoneapi.dll",
                "sycl8.dll",
                "mkl_sycl_blas.5.dll",
                "mkl_sycl_lapack.5.dll",
                "mkl_sycl_dft.5.dll",
                "mkl_sycl_sparse.5.dll",
                "ur_loader.dll",
                "ur_adapter_opencl.dll",
                "libmmd.dll",
            ):
                (library / name).write_bytes(name.encode("ascii"))
        return root

    def create_python(self) -> Path:
        root = self.root / "python"
        root.mkdir()
        for name in ("python.exe", "python312.dll", "python312.zip", "python312._pth"):
            (root / name).write_bytes(name.encode("ascii"))
        (root / "LICENSE.txt").write_text("python license", encoding="ascii")
        return root

    def create_runtime_licenses(self) -> Path:
        root = self.root / "intel-licenses"
        mkl = root / "onemkl-2025.2" / "licensing"
        mkl.mkdir(parents=True)
        (mkl / "license.txt").write_text("Intel runtime license", encoding="ascii")
        (mkl / "third-party-programs.txt").write_text("MKL notices", encoding="ascii")
        compiler = root / "dpcpp-compiler-runtime-2025.2" / "compiler"
        compiler.mkdir(parents=True)
        (compiler / "LICENSE.rtf").write_text("Intel compiler license", encoding="ascii")
        (compiler / "credist.txt").write_text("Redistributable files", encoding="ascii")
        (compiler / "third-party-programs.txt").write_text("Compiler notices", encoding="ascii")
        return root

    def test_minimal_excludes_python_runtime_and_writes_manifest(self) -> None:
        stage, archive = package_release.build_package(self.args("minimal"), self.script)

        self.assertIsNone(archive)
        self.assertFalse((stage / "python312.dll").exists())
        self.assertTrue((stage / "start_cyxwiz.bat").is_file())
        manifest = json.loads((stage / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual("minimal", manifest["package"]["profile"])
        self.assertEqual("1.2.3", manifest["package"]["version"])
        self.assertEqual([], manifest["package"]["arrayfire_backends"])
        self.assertTrue(all(item["sha256"] for item in manifest["components"]))

    def test_full_oneapi_copies_validated_runtime_closure(self) -> None:
        arrayfire = self.create_arrayfire(oneapi=True)
        python = self.create_python()
        args = self.args(
            "full",
            "--arrayfire-dir",
            str(arrayfire),
            "--python-dir",
            str(python),
            "--python-version",
            "3.12.8",
            "--intel-runtime-license-dir",
            str(self.create_runtime_licenses()),
            "--backends",
            "oneapi",
        )

        stage, _ = package_release.build_package(args, self.script)

        runtime = stage / "arrayfire" / "bin"
        self.assertFalse((stage / "python312.dll").exists())
        self.assertTrue((stage / "python" / "python312.dll").is_file())
        self.assertTrue((runtime / "afcpu.dll").is_file())
        self.assertTrue((runtime / "afoneapi.dll").is_file())
        self.assertTrue((runtime / "ur_loader.dll").is_file())
        manifest = json.loads((stage / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(["cpu", "oneapi"], manifest["package"]["arrayfire_backends"])
        self.assertEqual("3.10.0", manifest["dependency_versions"]["arrayfire"])
        intel_components = [
            item for item in manifest["components"] if item["path"].startswith("THIRD_PARTY_LICENSES/Intel/")
        ]
        self.assertTrue(intel_components)
        self.assertTrue(all(item["source"] == "intel-runtime-license" for item in intel_components))

    def test_full_oneapi_rejects_incomplete_runtime(self) -> None:
        arrayfire = self.create_arrayfire(oneapi=True)
        (arrayfire / "lib" / "ur_loader.dll").unlink()
        args = self.args(
            "full",
            "--arrayfire-dir",
            str(arrayfire),
            "--python-dir",
            str(self.create_python()),
            "--python-version",
            "3.12.8",
            "--intel-runtime-license-dir",
            str(self.create_runtime_licenses()),
            "--backends",
            "oneapi",
        )

        with self.assertRaisesRegex(package_release.PackageError, "Unified Runtime loader"):
            package_release.build_package(args, self.script)

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(package_release.PackageError, "Unsupported backend"):
            package_release.parse_backends("cpu,vulkan")

    def test_full_rejects_wrong_python_abi(self) -> None:
        with self.assertRaisesRegex(package_release.PackageError, "Python 3.12"):
            package_release.validate_python_version("3.14.0")

    def test_full_rejects_missing_runtime_notices(self) -> None:
        args = self.args(
            "full",
            "--arrayfire-dir",
            str(self.create_arrayfire()),
            "--python-dir",
            str(self.create_python()),
            "--python-version",
            "3.12.8",
        )
        with self.assertRaisesRegex(package_release.PackageError, "intel-runtime-license-dir"):
            package_release.build_package(args, self.script)

    def test_full_rejects_incomplete_intel_runtime_notices(self) -> None:
        notices = self.root / "incomplete-intel-notices"
        notices.mkdir()
        (notices / "LICENSE.txt").write_text("generic license", encoding="ascii")
        args = self.args(
            "full",
            "--arrayfire-dir",
            str(self.create_arrayfire()),
            "--python-dir",
            str(self.create_python()),
            "--python-version",
            "3.12.8",
            "--intel-runtime-license-dir",
            str(notices),
            "--backends",
            "cpu",
        )

        with self.assertRaisesRegex(package_release.PackageError, "oneMKL license"):
            package_release.build_package(args, self.script)

    def test_full_rejects_developer_python_tree(self) -> None:
        root = self.root / "developer-python"
        root.mkdir()
        (root / "python.exe").write_bytes(b"python")
        (root / "python312.dll").write_bytes(b"runtime")
        with self.assertRaisesRegex(package_release.PackageError, "standard-library archive"):
            package_release.validate_windows_embedded_python(root)

    def test_archive_version_cannot_escape_output_root(self) -> None:
        with self.assertRaisesRegex(package_release.PackageError, "Invalid CyxWiz"):
            package_release.validate_release_version("../release", "CyxWiz")

    def test_base_profile_emits_cpu_only_deterministic_artifact(self) -> None:
        arrayfire = self.create_arrayfire(opencl=True)
        args = self.artifact_args(
            "base",
            "--arrayfire-dir",
            str(arrayfire),
            "--python-dir",
            str(self.create_python()),
            "--python-version",
            "3.12.8",
            "--intel-runtime-license-dir",
            str(self.create_runtime_licenses()),
            "--generated-utc",
            "2026-08-13T20:00:00Z",
        )

        stage, archive = package_release.build_package(args, self.script)
        first_hash = package_release.sha256_file(archive)
        _, rebuilt = package_release.build_package(args, self.script)

        self.assertEqual(first_hash, package_release.sha256_file(rebuilt))
        self.assertTrue((stage / "arrayfire" / "bin" / "af.dll").is_file())
        self.assertTrue((stage / "arrayfire" / "bin" / "afcpu.dll").is_file())
        self.assertFalse((stage / "arrayfire" / "bin" / "afopencl.dll").exists())
        self.assertFalse((stage / "start_cyxwiz.bat").exists())
        manifest = json.loads(
            archive.with_suffix(".zip.manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual("base", manifest["signed"]["pack_kind"])
        self.assertEqual("cpu", manifest["signed"]["backend"])
        self.assertEqual([], manifest["signatures"])
        self.assertTrue(archive.with_suffix(".zip.signed.json").is_file())

    def test_opencl_pack_excludes_cpu_base_and_validates_signature(self) -> None:
        arrayfire = self.create_arrayfire(opencl=True)
        args = self.artifact_args(
            "pack",
            "--backend",
            "opencl",
            "--arrayfire-dir",
            str(arrayfire),
            "--generated-utc",
            "2026-08-13T20:00:00Z",
            "--signing-key-id",
            "release-2026",
            "--signature",
            "A" * 86,
        )

        stage, archive = package_release.build_package(args, self.script)

        self.assertTrue((stage / "runtime" / "afopencl.dll").is_file())
        self.assertFalse((stage / "runtime" / "af.dll").exists())
        self.assertFalse((stage / "runtime" / "afcpu.dll").exists())
        manifest = json.loads(
            archive.with_suffix(".zip.manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual("backend_pack", manifest["signed"]["pack_kind"])
        self.assertEqual("opencl", manifest["signed"]["backend"])
        package_release.validate_pack_manifest(manifest)

    def test_pack_profile_requires_one_optional_backend(self) -> None:
        args = self.args("pack", "--arrayfire-dir", str(self.create_arrayfire()))
        with self.assertRaisesRegex(package_release.PackageError, "requires exactly one"):
            package_release.build_package(args, self.script)

    def test_pack_stage_only_does_not_emit_archive(self) -> None:
        args = self.args(
            "pack",
            "--backend",
            "opencl",
            "--arrayfire-dir",
            str(self.create_arrayfire(opencl=True)),
        )

        stage, archive = package_release.build_package(args, self.script)

        self.assertTrue((stage / "runtime" / "afopencl.dll").is_file())
        self.assertIsNone(archive)

    def test_signing_rejects_tampered_canonical_input(self) -> None:
        args = self.artifact_args(
            "pack",
            "--backend",
            "opencl",
            "--arrayfire-dir",
            str(self.create_arrayfire(opencl=True)),
            "--generated-utc",
            "2026-08-13T20:00:00Z",
        )
        _, archive = package_release.build_package(args, self.script)
        assert archive is not None
        archive.with_suffix(".zip.signed.json").write_bytes(b"tampered")

        with self.assertRaisesRegex(sign_pack_manifest.SigningError, "does not match"):
            sign_pack_manifest.load_signature_input(
                archive.with_suffix(".zip.manifest.json")
            )

    def test_openssl_release_step_signs_and_self_verifies_manifest(self) -> None:
        openssl = shutil.which("openssl")
        if openssl is None:
            self.skipTest("OpenSSL is not available")
        args = self.artifact_args(
            "pack",
            "--backend",
            "opencl",
            "--arrayfire-dir",
            str(self.create_arrayfire(opencl=True)),
            "--generated-utc",
            "2026-08-13T20:00:00Z",
        )
        _, archive = package_release.build_package(args, self.script)
        assert archive is not None
        manifest_path = archive.with_suffix(".zip.manifest.json")
        private_key = self.root / "release-ed25519.pem"
        subprocess.run(
            [openssl, "genpkey", "-algorithm", "ED25519", "-out", str(private_key)],
            check=True,
            capture_output=True,
        )

        result = sign_pack_manifest.main(
            [
                str(manifest_path),
                "--private-key",
                str(private_key),
                "--key-id",
                "release-2026",
                "--openssl",
                openssl,
            ]
        )

        self.assertEqual(0, result)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        package_release.validate_pack_manifest(manifest)
        self.assertEqual("release-2026", manifest["signatures"][0]["key_id"])


if __name__ == "__main__":
    unittest.main()
