from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "python_runtime_package.py"
)
SPEC = importlib.util.spec_from_file_location("python_runtime_package", SCRIPT)
assert SPEC and SPEC.loader
python_runtime_package = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = python_runtime_package
SPEC.loader.exec_module(python_runtime_package)


class PythonRuntimePackageTests(unittest.TestCase):
    def test_policy_excludes_development_content_without_name_substrings(self) -> None:
        excluded = (
            "Lib/site-packages/demo/__pycache__/runtime.cpython-312.pyc",
            "Lib/site-packages/demo/tests/test_runtime.py",
            "Lib/test/test_json.py",
            ".pytest_cache/state",
        )
        retained = (
            "python.exe",
            "python312.zip",
            "Lib/site-packages/demo/runtime.py",
            "Lib/site-packages/demo/contest.py",
            "Lib/site-packages/demo/testing/probe.py",
            "Lib/site-packages/demo/testing_tools.py",
            "LICENSE.txt",
        )

        self.assertTrue(
            all(
                python_runtime_package.excludes_python_runtime_path(
                    PurePosixPath(path)
                )
                for path in excluded
            )
        )
        self.assertTrue(
            all(
                not python_runtime_package.excludes_python_runtime_path(
                    PurePosixPath(path)
                )
                for path in retained
            )
        )

    def test_copy_preserves_runtime_and_removes_tests_and_caches(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="cyxwiz-python-runtime-package-"
        ) as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            files = {
                "python.exe": b"runtime",
                "LICENSE.txt": b"license",
                "Lib/site-packages/demo/runtime.py": b"value = 1\n",
                "Lib/site-packages/demo/tests/test_runtime.py": b"test",
                "Lib/site-packages/demo/__pycache__/runtime.pyc": b"cache",
                "Lib/site-packages/demo/testing/probe.py": b"probe",
            }
            for relative, content in files.items():
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

            python_runtime_package.copy_python_runtime(source, destination)

            self.assertTrue((destination / "python.exe").is_file())
            self.assertTrue((destination / "LICENSE.txt").is_file())
            self.assertTrue(
                (destination / "Lib/site-packages/demo/runtime.py").is_file()
            )
            self.assertFalse(
                (destination / "Lib/site-packages/demo/tests").exists()
            )
            self.assertFalse(
                (destination / "Lib/site-packages/demo/__pycache__").exists()
            )
            self.assertTrue(
                (destination / "Lib/site-packages/demo/testing/probe.py").is_file()
            )


if __name__ == "__main__":
    unittest.main()
