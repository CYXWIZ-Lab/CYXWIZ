from __future__ import annotations

import unittest

from redist.scripts.backend_pack_target import (
    BackendPackTargetError,
    resolve_backend_pack_target,
)


class BackendPackTargetTests(unittest.TestCase):
    def test_windows_amd64_uses_signed_x86_64_identity(self) -> None:
        target = resolve_backend_pack_target("Windows", "AMD64")

        self.assertEqual("win64", target.platform)
        self.assertEqual("x86_64", target.architecture)
        self.assertEqual("win64-x86_64", target.artifact_suffix)
        self.assertEqual(".exe", target.executable_suffix)
        self.assertEqual(".dll", target.library_suffix)

    def test_macos_intel_and_arm_have_distinct_artifact_identities(self) -> None:
        intel = resolve_backend_pack_target("Darwin", "x86_64")
        arm = resolve_backend_pack_target("Darwin", "arm64")

        self.assertEqual("macos-x86_64", intel.artifact_suffix)
        self.assertEqual("macos-arm64", arm.artifact_suffix)
        self.assertEqual(".dylib", intel.library_suffix)

    def test_linux_aarch64_is_rejected_until_it_is_a_release_target(self) -> None:
        with self.assertRaisesRegex(
            BackendPackTargetError, "Unsupported release target"
        ):
            resolve_backend_pack_target("Linux", "aarch64")

    def test_unknown_machine_is_rejected_instead_of_mislabeled(self) -> None:
        with self.assertRaisesRegex(
            BackendPackTargetError, "Unsupported packaging host"
        ):
            resolve_backend_pack_target("Windows", "riscv64")


if __name__ == "__main__":
    unittest.main()
