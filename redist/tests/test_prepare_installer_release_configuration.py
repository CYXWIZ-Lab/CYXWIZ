from __future__ import annotations

import base64
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "prepare_installer_release_configuration.py"
)
SPEC = importlib.util.spec_from_file_location(
    "prepare_installer_release_configuration", SCRIPT
)
release_configuration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release_configuration
SPEC.loader.exec_module(release_configuration)


def trust_document(*roles: str, revoked: bool = False) -> dict[str, object]:
    return {
        "schema_version": 1,
        "keys": [
            {
                "key_id": "release-2026",
                "algorithm": "ed25519",
                "public_key": base64.urlsafe_b64encode(bytes(32))
                .decode("ascii")
                .rstrip("="),
                "roles": list(roles),
                "revoked": revoked,
            }
        ],
    }


class InstallerReleaseConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="cyxwiz-release-configuration-test-"
        )
        self.root = Path(self.temporary.name)
        self.trust = self.root / "input.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def args(self):
        return release_configuration.parse_args(
            [
                "--trust-store", str(self.trust),
                "--repository", "CYXWIZ-Lab/CYXWIZ",
                "--release-tag", "v0.2.0-alpha.1",
                "--cyxwiz-release", "0.2.0",
                "--bundle-version", "1.0.0",
                "--platform", "macos",
                "--architecture", "x86_64",
                "--trust-output", str(self.root / "out" / "trusted-keys.json"),
                "--configuration-output", str(self.root / "out" / "configuration.json"),
            ]
        )

    def write_trust(self, document: dict[str, object]) -> bytes:
        content = (json.dumps(document, indent=2) + "\n").encode("utf-8")
        self.trust.write_bytes(content)
        return content

    def test_materializes_exact_public_trust_and_descriptor_url(self) -> None:
        content = self.write_trust(
            trust_document("catalog", "pack", "installer")
        )
        result = release_configuration.prepare(self.args())
        self.assertEqual(
            "https://github.com/CYXWIZ-Lab/CYXWIZ/releases/download/"
            "v0.2.0-alpha.1/"
            "cyxwiz-installer-0.2.0-1.0.0-macos-x86_64.descriptor.json",
            result["descriptor_url"],
        )
        self.assertEqual(
            content, (self.root / "out" / "trusted-keys.json").read_bytes()
        )
        self.assertEqual(
            result,
            json.loads(
                (self.root / "out" / "configuration.json").read_text(
                    encoding="utf-8"
                )
            ),
        )

    def test_rejects_trust_without_active_installer_role(self) -> None:
        self.write_trust(trust_document("catalog", "pack"))
        with self.assertRaisesRegex(
            release_configuration.ReleaseConfigurationError,
            "active installer signing key",
        ):
            release_configuration.prepare(self.args())

    def test_rejects_revoked_installer_role(self) -> None:
        self.write_trust(trust_document("installer", revoked=True))
        with self.assertRaisesRegex(
            release_configuration.ReleaseConfigurationError,
            "active installer signing key",
        ):
            release_configuration.prepare(self.args())

    def test_rejects_unsafe_release_tag(self) -> None:
        self.write_trust(trust_document("installer"))
        arguments = self.args()
        arguments.release_tag = "../../latest"
        with self.assertRaisesRegex(
            release_configuration.ReleaseConfigurationError,
            "release tag",
        ):
            release_configuration.prepare(arguments)

    def test_rejects_unsupported_target(self) -> None:
        self.write_trust(trust_document("installer"))
        arguments = self.args()
        arguments.architecture = "armv7"
        with self.assertRaisesRegex(
            release_configuration.ReleaseConfigurationError,
            "unsupported",
        ):
            release_configuration.prepare(arguments)


if __name__ == "__main__":
    unittest.main()
