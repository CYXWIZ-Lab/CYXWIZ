from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "macho_runtime_closure.py"
SPEC = importlib.util.spec_from_file_location("macho_runtime_closure", SCRIPT)
assert SPEC and SPEC.loader
macho = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = macho
SPEC.loader.exec_module(macho)


class MachORuntimeClosureTests(unittest.TestCase):
    def test_dependency_parser_ignores_binary_header(self) -> None:
        output = (
            "/tmp/cyxwiz-engine:\n"
            "\t@rpath/libcyxwiz-backend.dylib (compatibility version 1.0.0, current version 1.0.0)\n"
            "\t/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 1.0.0)\n"
        )
        self.assertEqual(
            ("@rpath/libcyxwiz-backend.dylib", "/usr/lib/libc++.1.dylib"),
            macho.parse_otool_dependencies(output),
        )

    def test_rpath_parser_reads_only_lc_rpath_commands(self) -> None:
        output = (
            "Load command 1\n"
            "          cmd LC_RPATH\n"
            "      cmdsize 40\n"
            "         path @loader_path/../lib (offset 12)\n"
            "Load command 2\n"
            "          cmd LC_LOAD_DYLIB\n"
            "         name /tmp/not-an-rpath (offset 24)\n"
        )
        self.assertEqual(
            ("@loader_path/../lib",), macho.parse_otool_rpaths(output)
        )

    def test_only_apple_system_locations_are_host_dependencies(self) -> None:
        self.assertTrue(macho.is_system_dependency("/usr/lib/libc++.1.dylib"))
        self.assertTrue(
            macho.is_system_dependency(
                "/System/Library/Frameworks/Cocoa.framework/Versions/A/Cocoa"
            )
        )
        self.assertFalse(
            macho.is_system_dependency("/usr/local/opt/libomp/lib/libomp.dylib")
        )

    def test_macho_detection_excludes_executable_scripts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cyxwiz-macho-") as temporary:
            root = Path(temporary)
            script = root / "launch"
            script.write_bytes(b"#!/bin/sh\nexit 0\n")
            binary = root / "cyxwiz-engine"
            binary.write_bytes(b"\xcf\xfa\xed\xfe" + b"fixture")

            self.assertFalse(macho.is_macho(script))
            self.assertTrue(macho.is_macho(binary))

    def test_packaged_reference_is_relative_to_each_loader(self) -> None:
        owner = Path("/package/arrayfire/lib/libafcpu.dylib")
        dependency = Path("/package/lib/libomp.dylib")
        self.assertEqual(
            "@loader_path/../../lib/libomp.dylib",
            macho.packaged_reference(owner, dependency),
        )


if __name__ == "__main__":
    unittest.main()
