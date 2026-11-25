"""The programs a release ships never get the test scratch user folder.

cmake/CyxWizTestScratchUserFolders.cmake lists them (CYXWIZ_SHIPPED_EXECUTABLES)
and stops the configure when one matches the test rule; this test keeps that
list from drifting from what redist/scripts/package_release.py ships.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CMAKE_MODULE = REPO / "cmake" / "CyxWizTestScratchUserFolders.cmake"
PACKAGE_RELEASE = REPO / "redist" / "scripts" / "package_release.py"


def cmake_list(name: str) -> set[str]:
    text = CMAKE_MODULE.read_text(encoding="utf-8")
    match = re.search(rf"set\({name}\s+(.*?)\s+CACHE INTERNAL", text, re.S)
    if not match:
        raise AssertionError(f"{name} is not set in {CMAKE_MODULE.name}")
    return set(match.group(1).split())


def shipped_by_package_release() -> set[str]:
    text = PACKAGE_RELEASE.read_text(encoding="utf-8")
    # paths.build / f"cyxwiz-engine{exe_suffix}" ...: the programs the base copies from the build
    names = set(re.findall(r'paths\.build / f"(cyxwiz-[a-z0-9-]+)\{exe_suffix\}"', text))
    if not names:
        raise AssertionError("package_release.py names no program it ships")
    return names


def cmake_executables() -> set[str]:
    targets: set[str] = set()
    for path in list(REPO.glob("**/CMakeLists.txt")) + list((REPO / "cmake").glob("*.cmake")):
        if any(part.startswith(("build", "vcpkg", "third_party", "external")) for part in path.relative_to(REPO).parts):
            continue
        targets |= set(re.findall(r"add_executable\(\s*([A-Za-z0-9_.-]+)", path.read_text(encoding="utf-8",
                                                                                         errors="replace")))
    return targets


class ShippedExecutablesTest(unittest.TestCase):
    def test_every_shipped_program_is_listed(self) -> None:
        missing = shipped_by_package_release() - cmake_list("CYXWIZ_SHIPPED_EXECUTABLES")
        self.assertEqual(missing, set(), "add them to CYXWIZ_SHIPPED_EXECUTABLES")

    def test_listed_programs_are_targets_and_not_test_tools(self) -> None:
        shipped = cmake_list("CYXWIZ_SHIPPED_EXECUTABLES")
        tools = cmake_list("CYXWIZ_SCRATCH_TEST_TOOLS")
        self.assertEqual(shipped & tools, set())
        self.assertEqual(shipped - cmake_executables(), set(), "a listed program is no target any more")
        for name in tools:
            self.assertNotRegex(name, r"^(test|bench|benchmark|tool)_", "already covered by the name rule")


if __name__ == "__main__":
    unittest.main()
