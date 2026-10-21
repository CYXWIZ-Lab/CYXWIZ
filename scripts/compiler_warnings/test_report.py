import json
import tempfile
import unittest
from pathlib import Path

from scripts.compiler_warnings import report


class CompilerWarningReportTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path("/workspace/CYXWIZ")

    def test_classifies_clang_owned_external_and_toolchain_warnings(self):
        diagnostics = report.parse_diagnostics(
            [
                "[1/2] Building CXX object cyxwiz-engine/CMakeFiles/cyxwiz-engine.dir/src/main.cpp.o",
                "/workspace/CYXWIZ/cyxwiz-engine/src/main.cpp:12:4: warning: unused value [-Wunused-value]",
                "/usr/local/include/af/half.h:15:9: warning: anonymous struct [-Wgnu-anonymous-struct]",
                "ld: warning: ignoring duplicate libraries: 'libexample.a'",
            ],
            self.repo_root,
        )

        self.assertEqual([item.ownership for item in diagnostics], ["owned", "external", "toolchain"])
        self.assertEqual(diagnostics[0].target, "cyxwiz-engine")
        self.assertEqual(diagnostics[0].code, "-Wunused-value")

    def test_classifies_msvc_warning_and_uses_project_as_target(self):
        diagnostics = report.parse_diagnostics(
            [
                r"C:\a\CYXWIZ\cyxwiz-backend\src\tensor.cpp(21,7): warning C4267: conversion [C:\a\build\cyxwiz-backend.vcxproj]"
            ],
            self.repo_root,
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].ownership, "owned")
        self.assertEqual(diagnostics[0].source, "cyxwiz-backend/src/tensor.cpp")
        self.assertEqual(diagnostics[0].target, "cyxwiz-backend")
        self.assertEqual(diagnostics[0].code, "C4267")

    def test_normalizes_repeated_github_checkout_name_and_deduplicates_msvc_details(self):
        diagnostics = report.parse_diagnostics(
            [
                r"D:\a\CYXWIZ\CYXWIZ\cyxwiz-backend\src\text.cpp(12,5): warning C4244: conversion [D:\a\CYXWIZ\CYXWIZ\build\cyxwiz-backend.vcxproj]",
                r"D:\a\CYXWIZ\CYXWIZ\cyxwiz-backend\src\text.cpp(12,5): warning C4244: with [D:\a\CYXWIZ\CYXWIZ\build\cyxwiz-backend.vcxproj]",
            ],
            self.repo_root,
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].source, "cyxwiz-backend/src/text.cpp")
        self.assertEqual(diagnostics[0].ownership, "owned")

    def test_classifies_msvc_tool_warning_without_source(self):
        diagnostics = report.parse_diagnostics(
            [
                r"LINK : warning LNK4099: missing PDB [C:\a\build\cyxwiz-engine.vcxproj]"
            ],
            self.repo_root,
        )

        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0].ownership, "toolchain")
        self.assertEqual(diagnostics[0].target, "cyxwiz-engine")
        self.assertEqual(diagnostics[0].code, "LNK4099")

    def test_dependency_checkout_and_vendored_source_are_not_owned(self):
        diagnostics = report.parse_diagnostics(
            [
                "/workspace/CYXWIZ/build/vcpkg_installed/include/lib.h:1:2: warning: dependency warning [-Wpedantic]",
                "/workspace/CYXWIZ/cyxwiz-engine/src/plugin/security/tweetnacl.c:2:3: warning: vendored warning [-Wunused-parameter]",
            ],
            self.repo_root,
        )

        self.assertEqual([item.ownership for item in diagnostics], ["generated", "vendored"])

    def test_report_fails_only_when_owned_limit_is_exceeded(self):
        diagnostic = report.Diagnostic(
            source="cyxwiz-engine/src/main.cpp",
            line=1,
            column=None,
            code="-Wunused-variable",
            message="unused variable",
            target="cyxwiz-engine",
            ownership="owned",
        )

        passing = report.build_report([], "macos", owned_limit=0)
        failing = report.build_report([diagnostic], "macos", owned_limit=0)

        self.assertTrue(passing["passed"])
        self.assertFalse(failing["passed"])
        self.assertIn("cyxwiz-engine/src/main.cpp:1", report.render_markdown(failing))

    def test_loads_platform_limit_and_falls_back_to_default(self):
        with tempfile.TemporaryDirectory() as directory:
            baseline = Path(directory) / "baseline.json"
            baseline.write_text(
                json.dumps({"owned_warning_limit": {"default": 3, "linux": 1}}),
                encoding="utf-8",
            )

            self.assertEqual(report.load_owned_limit(baseline, "linux"), 1)
            self.assertEqual(report.load_owned_limit(baseline, "macos"), 3)


if __name__ == "__main__":
    unittest.main()
