import tempfile
from pathlib import Path
import unittest

from redist.scripts.normalize_arrayfire_input import discover, normalize


class ArrayFireInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="cyxwiz-af-input-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "source"
        (self.root / "include" / "af").mkdir(parents=True)
        (self.root / "include" / "af" / "version.h").write_text("fixture")
        (self.root / "lib64").mkdir()
        for name in ("libaf.so.3", "libafcpu.so.3"):
            (self.root / "lib64" / name).write_bytes(b"fixture")
        self.destination = Path(self.temporary.name) / "stage"

    def test_linux_copies_all_notices_without_license_filename_prefix(self):
        notices = self.root / "LICENSES"
        notices.mkdir()
        for name in ("Apache-2.0.txt", "BSD 3-Clause.txt"):
            (notices / name).write_text(name)
        normalize(self.root, self.destination)
        self.assertEqual(sorted(p.name for p in (self.destination / "LICENSES").iterdir()),
                         ["Apache-2.0.txt", "BSD 3-Clause.txt"])
        self.assertTrue((self.destination / "lib" / "libafcpu.so.3").is_file())

    def test_macos_root_license_and_dylibs(self):
        for item in (self.root / "lib64").iterdir():
            item.unlink()
        for name in ("libaf.3.dylib", "libafcpu.3.dylib"):
            (self.root / "lib64" / name).write_bytes(b"fixture")
        (self.root / "LICENSE").write_text("notice")
        normalize(self.root, self.destination)
        self.assertEqual((self.destination / "LICENSES" / "LICENSE").read_text(), "notice")

    def test_missing_notices_fails_before_staging(self):
        (self.root / "LICENSES").mkdir()
        with self.assertRaisesRegex(ValueError, "Missing ArrayFire notices"):
            normalize(self.root, self.destination)
        self.assertFalse(self.destination.exists())

    def test_missing_cpu_library_is_explicit(self):
        (self.root / "lib64" / "libafcpu.so.3").unlink()
        with self.assertRaisesRegex(ValueError, "unified/CPU libraries"):
            discover(self.root)

    def test_empty_lib_does_not_hide_valid_lib64(self):
        (self.root / "lib").mkdir()
        (self.root / "LICENSE.txt").write_text("notice")
        libraries, _ = discover(self.root)
        self.assertEqual(libraries, (self.root / "lib64").resolve())


if __name__ == "__main__":
    unittest.main()
