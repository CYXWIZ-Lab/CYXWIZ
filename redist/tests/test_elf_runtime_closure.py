from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "elf_runtime_closure.py"
SPEC = importlib.util.spec_from_file_location("elf_runtime_closure", SCRIPT)
assert SPEC and SPEC.loader
elf = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = elf
SPEC.loader.exec_module(elf)


class ElfRuntimeClosureTests(unittest.TestCase):
    def test_ldd_parser_preserves_missing_and_direct_dependencies(self) -> None:
        output = (
            "linux-vdso.so.1 (0x00007fff)\n"
            "libaf.so.3 => /opt/arrayfire/lib/libaf.so.3 (0x00007f00)\n"
            "libmissing.so.1 => not found\n"
            "/lib64/ld-linux-x86-64.so.2 (0x00007f01)\n"
        )
        dependencies = elf.parse_ldd_dependencies(output)
        self.assertEqual(
            ("libaf.so.3", "libmissing.so.1", "ld-linux-x86-64.so.2"),
            tuple(item.name for item in dependencies),
        )
        self.assertIsNone(dependencies[1].path)

    def test_system_library_policy_excludes_opt_and_usr_local(self) -> None:
        self.assertTrue(elf.is_linux_system_library(Path("/usr/lib/libX11.so.6")))
        self.assertTrue(elf.is_linux_system_library(Path("/lib/libc.so.6")))
        self.assertFalse(
            elf.is_linux_system_library(Path("/opt/arrayfire/lib/libaf.so.3"))
        )
        self.assertFalse(elf.is_linux_system_library(Path("/usr/local/lib/libvendor.so")))

    def test_packaged_rpath_reaches_root_and_runtime_from_nested_library(self) -> None:
        stage = Path("/package")
        value = elf.packaged_rpath(
            stage / "arrayfire" / "lib" / "libafcpu.so.3", stage
        )
        self.assertEqual(
            "$ORIGIN:$ORIGIN/../..:$ORIGIN/../../lib",
            value,
        )

    def test_closure_copies_non_system_dependency_and_patches_each_binary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="cyxwiz-elf-") as temporary:
            root = Path(temporary)
            stage = root / "stage"
            vendor = root / "vendor"
            stage.mkdir()
            vendor.mkdir()
            engine = stage / "cyxwiz-engine"
            dependency = vendor / "libvendor.so.1"
            engine.write_bytes(b"\x7fELFengine")
            dependency.write_bytes(b"\x7fELFvendor")
            commands: list[tuple[str, ...]] = []

            def runner(command):
                command = tuple(command)
                commands.append(command)
                if (
                    command[0] == "ldd"
                    and Path(command[1]).resolve() == engine.resolve()
                ):
                    output = f"libvendor.so.1 => {dependency} (0x01)\n"
                elif command[0] == "ldd":
                    output = "libc.so.6 => /usr/lib/libc.so.6 (0x02)\n"
                else:
                    output = ""
                return subprocess.CompletedProcess(command, 0, output)

            copied = elf.close_linux_runtime(stage, runner=runner)

            packaged = stage / "lib" / "libvendor.so.1"
            self.assertEqual([packaged.resolve()], copied)
            self.assertTrue(packaged.is_file())
            patched = [command for command in commands if command[0] == "patchelf"]
            self.assertEqual(2, len(patched))

    def test_closure_rejects_unresolved_dependency(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cyxwiz-elf-") as temporary:
            stage = Path(temporary)
            binary = stage / "cyxwiz-engine"
            binary.write_bytes(b"\x7fELFfixture")

            def runner(command):
                return subprocess.CompletedProcess(
                    command, 0, "libmissing.so.1 => not found\n"
                )

            with self.assertRaisesRegex(elf.ElfClosureError, "libmissing.so.1"):
                elf.close_linux_runtime(stage, runner=runner)


if __name__ == "__main__":
    unittest.main()
