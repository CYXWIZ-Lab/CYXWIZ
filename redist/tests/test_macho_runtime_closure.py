from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
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

    def test_resolves_transitive_dependency_beside_original_binary(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cyxwiz-macho-") as temporary:
            root = Path(temporary)
            source = root / "gcc" / "libgfortran.5.dylib"
            dependency = source.parent / "libquadmath.0.dylib"
            packaged = root / "package" / "lib" / source.name
            source.parent.mkdir(parents=True)
            packaged.parent.mkdir(parents=True)
            source.write_bytes(b"source")
            dependency.write_bytes(b"dependency")
            packaged.write_bytes(b"packaged")

            resolved = macho._resolve_dependency(
                "@rpath/libquadmath.0.dylib",
                macho.PackagedBinary(packaged, source),
                (),
                (),
            )

            self.assertEqual(dependency.resolve(), resolved)

    def test_plugin_sdk_copy_of_a_dylib_is_not_a_duplicate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cyxwiz-macho-") as temporary:
            stage = Path(temporary)
            (stage / "sdk" / "lib").mkdir(parents=True)
            engine = stage / "cyxwiz-engine"
            backend = stage / "libcyxwiz-backend.dylib"
            sdk_backend = stage / "sdk" / "lib" / "libcyxwiz-backend.dylib"
            for path in (engine, backend, sdk_backend):
                path.write_bytes(b"\xcf\xfa\xed\xfe" + str(path).encode())
            commands: list[tuple[str, ...]] = []

            def runner(command):
                command = tuple(command)
                commands.append(command)
                output = ""
                if command[:2] == ("otool", "-L"):
                    output = f"{command[-1]}:\n"
                    if Path(command[-1]) == engine.resolve():
                        output += "\t@rpath/libcyxwiz-backend.dylib (compatibility version 0.0.0)\n"
                return subprocess.CompletedProcess(command, 0, output)

            copied = macho.close_macos_runtime(stage, (), runner=runner)

            self.assertEqual([], copied)
            ids = {Path(command[-1]) for command in commands if command[:2] == ("install_name_tool", "-id")}
            self.assertEqual({backend.resolve(), sdk_backend.resolve()}, ids)
            # The Engine's dependency goes to the base copy, never to the SDK's
            changes = [command for command in commands if command[:2] == ("install_name_tool", "-change")]
            self.assertEqual([("install_name_tool", "-change", "@rpath/libcyxwiz-backend.dylib",
                               "@loader_path/libcyxwiz-backend.dylib", str(engine.resolve()))], changes)

    def test_two_copies_inside_the_plugin_sdk_are_refused(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cyxwiz-macho-") as temporary:
            stage = Path(temporary)
            (stage / "sdk" / "lib").mkdir(parents=True)
            (stage / "sdk" / "bin").mkdir(parents=True)
            for path in (stage / "sdk" / "lib" / "libtwice.dylib", stage / "sdk" / "bin" / "libtwice.dylib"):
                path.write_bytes(b"\xcf\xfa\xed\xfe" + str(path).encode())

            def runner(command):
                return subprocess.CompletedProcess(tuple(command), 0, "")

            with self.assertRaisesRegex(macho.MachOClosureError, "Duplicate packaged Mach-O name libtwice.dylib"):
                macho.close_macos_runtime(stage, (), runner=runner)

    def test_two_runtime_copies_of_one_dylib_name_are_refused(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cyxwiz-macho-") as temporary:
            stage = Path(temporary)
            (stage / "lib").mkdir()
            for path in (stage / "libtwice.dylib", stage / "lib" / "libtwice.dylib"):
                path.write_bytes(b"\xcf\xfa\xed\xfe" + str(path).encode())

            def runner(command):
                return subprocess.CompletedProcess(tuple(command), 0, "")

            with self.assertRaisesRegex(macho.MachOClosureError, "Duplicate packaged Mach-O name libtwice.dylib"):
                macho.close_macos_runtime(stage, (), runner=runner)

    def test_build_machine_rpaths_are_foreign(self) -> None:
        stage = Path("/package")
        engine = stage / "cyxwiz-engine"
        libaf = stage / "arrayfire" / "lib" / "libaf.3.dylib"
        self.assertEqual(
            [
                "/usr/local/opt/arrayfire/lib",
                "/Users/runner/work/CYXWIZ/CYXWIZ/build/lib",
                "@executable_path/../Frameworks",
            ],
            macho.foreign_rpaths(
                engine,
                (
                    "/usr/local/opt/arrayfire/lib",
                    "/Users/runner/work/CYXWIZ/CYXWIZ/build/lib",
                    "@loader_path/lib",
                    "@executable_path/../Frameworks",
                ),
                stage,
            ),
        )
        # Homebrew bottle entries resolve above the package root.
        self.assertEqual(
            ["@loader_path/../../../../lib"],
            macho.foreign_rpaths(
                libaf, ("@loader_path/../../../../lib", "@loader_path/../../lib"), stage
            ),
        )


if __name__ == "__main__":
    unittest.main()
