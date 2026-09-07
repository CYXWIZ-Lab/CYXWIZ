from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/verify_data_convert_package.py"
SPEC = importlib.util.spec_from_file_location("verify_data_convert_package", SCRIPT)
assert SPEC and SPEC.loader
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


class DataConvertPackageTests(unittest.TestCase):
    def test_import_parser(self):
        self.assertEqual(verifier.imported_dlls(
            "Dump of file example.dll\n    archive.dll\n    nowide.dll\n    archive.dll\n"),
            ["archive.dll", "nowide.dll"])

    def test_recursive_closure_and_missing_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage = root / "stage"
            stage.mkdir()
            (stage / "archive.dll").write_bytes(b"archive")
            (root / "System32").mkdir()
            (root / "System32/KERNEL32.dll").write_bytes(b"system")

            def inspect(command, **kwargs):
                output = ("    archive.dll\n" if command[-1].endswith("test.exe")
                          else "    zlib1.dll\n    KERNEL32.dll\n")
                return subprocess.CompletedProcess(command, 0, output, "")

            with patch.dict(os.environ, {"SystemRoot": str(root)}), \
                 patch.object(verifier.subprocess, "run", side_effect=inspect):
                with self.assertRaisesRegex(RuntimeError, "Missing staged dependency.*zlib1"):
                    verifier.audit_dependencies([root / "test.exe"], stage, Path("dumpbin"))
                (stage / "zlib1.dll").write_bytes(b"zlib")
                evidence = verifier.audit_dependencies([root / "test.exe"], stage, Path("dumpbin"))
            self.assertEqual(set(evidence["bundled_dll_sha256"]), {"archive.dll", "zlib1.dll"})
            self.assertEqual(evidence["system_prerequisites"], ["KERNEL32.dll"])

    def test_debug_crt_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name in ("MSVCP140D.dll", "VCRUNTIME140_1D.dll", "ucrtbased.dll"):
                output = subprocess.CompletedProcess([], 0, "    " + name + "\n", "")
                with patch.dict(os.environ, {"SystemRoot": temporary}), \
                     patch.object(verifier.subprocess, "run", return_value=output):
                    with self.assertRaisesRegex(RuntimeError, "Debug CRT"):
                        verifier.audit_dependencies([Path(temporary) / "test.exe"],
                                                    Path(temporary), Path("dumpbin"))


if __name__ == "__main__":
    unittest.main()
