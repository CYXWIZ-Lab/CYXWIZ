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
            "$ORIGIN:$ORIGIN/../..:$ORIGIN/../../lib:$ORIGIN/../../python/lib",
            value,
        )

    def test_engine_rpath_reaches_bundled_libpython(self) -> None:
        stage = Path("/package")
        self.assertEqual(
            "$ORIGIN:$ORIGIN/lib:$ORIGIN/arrayfire/lib:$ORIGIN/python/lib",
            elf.packaged_rpath(stage / "cyxwiz-engine", stage),
        )
        self.assertTrue(elf.is_bundled_python(stage / "python" / "bin" / "python3.12", stage))
        self.assertFalse(elf.is_bundled_python(stage / "lib" / "libpython3.12.so.1.0", stage))

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

    def test_plugin_sdk_copy_of_a_library_is_not_a_duplicate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cyxwiz-elf-") as temporary:
            root = Path(temporary)
            stage = root / "stage"
            build = root / "build"
            (stage / "sdk" / "lib").mkdir(parents=True)
            build.mkdir()
            engine = stage / "cyxwiz-engine"
            backend = stage / "libcyxwiz-backend.so"
            sdk_backend = stage / "sdk" / "lib" / "libcyxwiz-backend.so"
            built_backend = build / "libcyxwiz-backend.so"
            for path in (engine, backend, sdk_backend, built_backend):
                path.write_bytes(b"\x7fELF" + path.name.encode())
            commands: list[tuple[str, ...]] = []

            def runner(command):
                command = tuple(command)
                commands.append(command)
                output = ""
                if command[0] == "ldd" and Path(command[1]).resolve() == engine.resolve():
                    output = f"libcyxwiz-backend.so => {built_backend} (0x01)\n"
                elif command[0] == "ldd":
                    output = "libc.so.6 => /usr/lib/libc.so.6 (0x02)\n"
                return subprocess.CompletedProcess(command, 0, output)

            copied = elf.close_linux_runtime(stage, runner=runner)

            # The Engine's dependency is the base copy: nothing copied, the SDK's never chosen
            self.assertEqual([], copied)
            self.assertFalse((stage / "lib").exists())
            patched = {Path(command[-1]) for command in commands if command[0] == "patchelf"}
            self.assertEqual({engine.resolve(), backend.resolve(), sdk_backend.resolve()}, patched)

    def test_two_runtime_copies_of_one_name_are_refused(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cyxwiz-elf-") as temporary:
            stage = Path(temporary)
            (stage / "lib").mkdir()
            for path in (stage / "libtwice.so", stage / "lib" / "libtwice.so"):
                path.write_bytes(b"\x7fELF" + str(path).encode())

            def runner(command):
                return subprocess.CompletedProcess(command, 0, "")

            with self.assertRaisesRegex(elf.ElfClosureError, "Duplicate packaged ELF name libtwice.so"):
                elf.close_linux_runtime(stage, runner=runner)

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

    def test_relocated_binary_resolves_dependency_from_explicit_input_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cyxwiz-elf-") as temporary:
            root = Path(temporary).resolve()
            stage, vendor = root / "stage", root / "vendor"
            stage.mkdir()
            vendor.mkdir()
            (stage / "cyxwiz-engine").write_bytes(b"\x7fELFengine")
            dependency = vendor / "libmkl_rt.so.2"
            dependency.write_bytes(b"\x7fELFmkl")
            inspections = []

            def runner(command):
                if command[0] == "env":
                    inspections.append(command)
                    self.assertEqual("ldd", command[2])
                    self.assertIn(str(vendor), command[1])
                    output = (f"libmkl_rt.so.2 => {dependency} (0x01)\n"
                              if Path(command[3]).name == "cyxwiz-engine" else "")
                else:
                    self.assertEqual("patchelf", command[0])
                    output = ""
                return subprocess.CompletedProcess(command, 0, output)

            copied = elf.close_linux_runtime(stage, search_roots=[vendor], runner=runner)
            self.assertEqual([stage / "lib" / "libmkl_rt.so.2"], copied)
            self.assertEqual(2, len(inspections))


if __name__ == "__main__":
    unittest.main()
