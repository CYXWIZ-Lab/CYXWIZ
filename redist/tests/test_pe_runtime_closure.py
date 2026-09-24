from __future__ import annotations

import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "pe_runtime_closure.py"
SPEC = importlib.util.spec_from_file_location("pe_runtime_closure", SCRIPT)
assert SPEC and SPEC.loader
pe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pe
SPEC.loader.exec_module(pe)


def synthetic_pe(imports: list[str]) -> bytes:
    """A minimal PE32+ image whose only content is an import directory."""
    image = bytearray(0x400)
    image[0:2] = b"MZ"
    struct.pack_into("<I", image, 0x3C, 0x40)
    image[0x40:0x44] = b"PE\0\0"
    optional_size = 112 + 16 * 8
    struct.pack_into("<HHIIIHH", image, 0x44, 0x8664, 1, 0, 0, 0, optional_size, 0x22)
    optional = 0x58
    struct.pack_into("<H", image, optional, 0x20B)
    section_rva, section_raw = 0x1000, 0x200
    struct.pack_into("<II", image, optional + 112 + 8, section_rva, 20 * (len(imports) + 1))
    section = optional + optional_size
    image[section:section + 8] = b".idata\0\0"
    struct.pack_into("<IIII", image, section + 8, 0x200, section_rva, 0x200, section_raw)
    names = section_raw + 20 * (len(imports) + 1)
    for index, name in enumerate(imports):
        descriptor = section_raw + 20 * index
        struct.pack_into("<I", image, descriptor, 1)
        struct.pack_into("<I", image, descriptor + 12, section_rva + names - section_raw)
        encoded = name.encode("ascii") + b"\0"
        image[names:names + len(encoded)] = encoded
        names += len(encoded)
    return bytes(image)


class PeRuntimeClosureTests(unittest.TestCase):
    def test_reads_import_directory_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tool.exe"
            path.write_bytes(synthetic_pe(["KERNEL32.dll", "VCRUNTIME140.dll"]))
            self.assertTrue(pe.is_pe(path))
            self.assertEqual(
                ["KERNEL32.dll", "VCRUNTIME140.dll"], pe.pe_imported_dlls(path)
            )

    def test_non_pe_files_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "notes.dll"
            path.write_bytes(b"not a portable executable")
            self.assertFalse(pe.is_pe(path))
            self.assertEqual([], pe.audit_msvc_runtime_closure(Path(temporary)))

    def test_executable_needs_runtime_in_its_own_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "runtime").mkdir()
            (root / "launcher.exe").write_bytes(synthetic_pe(["MSVCP140.dll"]))
            (root / "runtime" / "engine.exe").write_bytes(
                synthetic_pe(["VCRUNTIME140.dll", "MSVCP140.dll"])
            )
            (root / "runtime" / "msvcp140.dll").write_bytes(b"x")
            problems = pe.audit_msvc_runtime_closure(root)
            self.assertEqual(
                [
                    "launcher.exe needs msvcp140.dll",
                    "runtime/engine.exe needs vcruntime140.dll",
                ],
                problems,
            )

    def test_library_may_use_runtime_shipped_elsewhere_in_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "arrayfire" / "bin").mkdir(parents=True)
            (root / "arrayfire" / "bin" / "af.dll").write_bytes(
                synthetic_pe(["VCRUNTIME140.dll", "KERNEL32.dll"])
            )
            (root / "engine.exe").write_bytes(synthetic_pe(["VCRUNTIME140.dll"]))
            (root / "VCRUNTIME140.dll").write_bytes(b"x")
            self.assertEqual([], pe.audit_msvc_runtime_closure(root))


if __name__ == "__main__":
    unittest.main()
