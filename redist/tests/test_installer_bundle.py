from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
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


contract = load_script("installer_bundle_contract")
packager = load_script("package_installer_bundle")
signer = load_script("sign_installer_bundle")


class InstallerBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="cyxwiz-installer-bundle-")
        self.root = Path(self.temporary.name)
        self.stage = self.root / "stage"
        for relative, content in {
            "cyxwiz-installer.exe": b"installer",
            "cyxwiz-backend-pack-installer.exe": b"helper",
            "resources/cyxwiz.png": b"logo",
            "runtime/trust/trusted-keys.json": b"{}\n",
            "runtime/catalogs/current.json": b"{}\n",
        }.items():
            path = self.stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            if path.suffix == ".exe":
                path.chmod(0o755)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def arguments(self, output: Path) -> list[str]:
        return [
            str(self.stage),
            str(output),
            "--bundle-version", "1.0.0",
            "--cyxwiz-release", "0.2.0",
            "--release-channel", "alpha",
            "--platform", "windows",
            "--architecture", "x86_64",
            "--minimum-setup-version", "1.0.0",
            "--generated-utc", "2026-08-27T00:00:00Z",
            "--expires-utc", "2026-11-27T00:00:00Z",
        ]

    def package(self, output: Path):
        args = packager.parse_args(self.arguments(output))
        return packager.run(args)

    def test_bundle_is_deterministic_and_exactly_inventoried(self) -> None:
        first = self.package(self.root / "first")
        second = self.package(self.root / "second")
        self.assertEqual(first[0].read_bytes(), second[0].read_bytes())
        self.assertEqual(first[1].read_bytes(), second[1].read_bytes())

        descriptor = json.loads(first[1].read_text(encoding="utf-8"))
        contract.validate_installer_bundle_descriptor(
            descriptor, require_signature=False
        )
        inventory = descriptor["signed"]["components"]
        self.assertEqual(
            [entry["path"] for entry in inventory],
            sorted(path.relative_to(self.stage).as_posix()
                   for path in self.stage.rglob("*") if path.is_file()),
        )
        with zipfile.ZipFile(first[0]) as archive:
            self.assertEqual(archive.namelist(), [entry["path"] for entry in inventory])
            for entry in inventory:
                self.assertEqual(
                    hashlib.sha256(archive.read(entry["path"])).hexdigest(),
                    entry["sha256"],
                )

    def test_missing_bootstrap_file_and_unknown_schema_field_fail(self) -> None:
        (self.stage / "runtime/catalogs/current.json").unlink()
        missing_output = self.root / "missing"
        with self.assertRaisesRegex(ValueError, "required bootstrap files"):
            self.package(missing_output)
        self.assertEqual(list(missing_output.iterdir()), [])

        (self.stage / "runtime/catalogs/current.json").write_bytes(b"{}\n")
        valid_output = self.root / "valid"
        _, descriptor_path, _ = self.package(valid_output)
        with self.assertRaisesRegex(
            packager.InstallerBundlePackagingError, "already exists"
        ):
            self.package(valid_output)
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
        expired = json.loads(json.dumps(descriptor))
        expired["signed"]["expires_utc"] = "2026-01-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "expiry must follow"):
            contract.validate_installer_bundle_descriptor(
                expired, require_signature=False
            )
        nonexecutable = json.loads(json.dumps(descriptor))
        next(
            entry
            for entry in nonexecutable["signed"]["components"]
            if entry["path"] == "cyxwiz-installer.exe"
        )["executable"] = False
        with self.assertRaisesRegex(ValueError, "entry points are not executable"):
            contract.validate_installer_bundle_descriptor(
                nonexecutable, require_signature=False
            )
        unordered = json.loads(json.dumps(descriptor))
        unordered["signed"]["components"].reverse()
        with self.assertRaisesRegex(ValueError, "inventory is not canonical"):
            contract.validate_installer_bundle_descriptor(
                unordered, require_signature=False
            )
        descriptor["signed"]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "unknown or missing"):
            contract.validate_installer_bundle_descriptor(
                descriptor, require_signature=False
            )

    def test_links_and_output_inside_stage_fail(self) -> None:
        with self.assertRaisesRegex(
            packager.InstallerBundlePackagingError, "output must be outside"
        ):
            self.package(self.stage / "output")
        link = self.stage / "linked-installer.exe"
        try:
            link.symlink_to(self.stage / "cyxwiz-installer.exe")
        except OSError:
            self.skipTest("symbolic links are unavailable")
        with self.assertRaisesRegex(
            packager.InstallerBundlePackagingError, "contains a link"
        ):
            self.package(self.root / "linked")

    def test_signer_binds_canonical_descriptor_body(self) -> None:
        openssl = shutil.which("openssl")
        if not openssl:
            self.skipTest("OpenSSL is unavailable")
        _, descriptor_path, signature_input = self.package(self.root / "signed")
        private_key = self.root / "installer.pem"
        public_key = self.root / "installer-public.pem"
        subprocess.run(
            [openssl, "genpkey", "-algorithm", "ED25519", "-out", str(private_key)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [openssl, "pkey", "-in", str(private_key), "-pubout", "-out", str(public_key)],
            check=True,
            capture_output=True,
        )
        original_signature_input = signature_input.read_bytes()
        signature_input.write_bytes(original_signature_input + b"tampered")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(
                signer.main(
                    [
                        str(descriptor_path),
                        "--private-key", str(private_key),
                        "--key-id", "installer-2026",
                        "--openssl", openssl,
                    ]
                ),
                1,
            )
        signature_input.write_bytes(original_signature_input)
        self.assertEqual(
            signer.main(
                [
                    str(descriptor_path),
                    "--private-key", str(private_key),
                    "--key-id", "installer-2026",
                    "--openssl", openssl,
                ]
            ),
            0,
        )
        document = json.loads(descriptor_path.read_text(encoding="utf-8"))
        contract.validate_installer_bundle_descriptor(document)
        signature = self.root / "signature.bin"
        import base64

        signature.write_bytes(
            base64.urlsafe_b64decode(document["signatures"][0]["value"] + "==")
        )
        subprocess.run(
            [
                openssl,
                "pkeyutl",
                "-verify",
                "-rawin",
                "-pubin",
                "-inkey", str(public_key),
                "-in", str(signature_input),
                "-sigfile", str(signature),
            ],
            check=True,
            capture_output=True,
        )


if __name__ == "__main__":
    unittest.main()
