"""Check release setup rejection without shell-native exit-code propagation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch


def verify_rejection(setup: Path, trust: Path) -> None:
    result = subprocess.run(
        [str(setup.resolve()), "--trust-store", str(trust),
         "--descriptor", "missing-descriptor.json"],
        capture_output=True, text=True, timeout=30, check=False,
    )
    output = result.stdout + result.stderr
    if result.returncode != 4 or "--trust-store is disabled" not in output:
        raise RuntimeError(
            f"Expected explicit trust override rejection (exit 4); "
            f"observed exit {result.returncode}:\n{output}"
        )


class ReleaseTrustCheckTests(unittest.TestCase):
    def check_result(self, code: int, stdout: str, stderr: str) -> None:
        with patch.object(subprocess, "run", return_value=subprocess.CompletedProcess(
            [], code, stdout, stderr
        )) as run:
            verify_rejection(Path("setup"), Path("trust.json"))
            self.assertEqual(run.call_args.kwargs["timeout"], 30)

    def test_accepts_expected_rejection_on_stderr(self) -> None:
        self.check_result(4, "", "--trust-store is disabled")

    def test_accepts_expected_rejection_on_stdout(self) -> None:
        self.check_result(4, "--trust-store is disabled", "")

    def test_rejects_success_and_unexpected_failures(self) -> None:
        for code in (0, 1, -11):
            with self.subTest(code=code), self.assertRaisesRegex(RuntimeError, f"observed exit {code}"):
                self.check_result(code, "", "--trust-store is disabled")

    def test_rejects_wrong_reason_and_preserves_diagnostics(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing descriptor"):
            self.check_result(4, "", "missing descriptor")

    def test_timeout_is_not_accepted(self) -> None:
        with patch.object(subprocess, "run", side_effect=subprocess.TimeoutExpired("setup", 30)):
            with self.assertRaises(subprocess.TimeoutExpired):
                verify_rejection(Path("setup"), Path("trust.json"))

    def test_missing_executable_is_not_accepted(self) -> None:
        with patch.object(subprocess, "run", side_effect=FileNotFoundError("missing setup")):
            with self.assertRaises(FileNotFoundError):
                verify_rejection(Path("setup"), Path("trust.json"))


if __name__ == "__main__":
    if "--setup" not in sys.argv:
        unittest.main()
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--setup", required=True, type=Path)
        args = parser.parse_args()
        try:
            verify_rejection(args.setup, Path(os.environ["CYXWIZ_RELEASE_TRUST_STORE"]))
        except (KeyError, OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            print(f"FAIL: {error}", file=sys.stderr)
            sys.exit(1)
        print("PASS: release setup explicitly rejects replacement trust (exit 4)")
