from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import package_installer_bundle as packager
import sign_installer_bundle as signer


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--platform", choices=("windows", "linux", "macos"), required=True)
    parser.add_argument("--architecture", choices=("x86_64", "arm64"), required=True)
    parser.add_argument("--openssl", required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    with tempfile.TemporaryDirectory(prefix="cyxwiz-setup-contract-") as temporary:
        root = Path(temporary)
        stage = root / "stage"
        suffix = ".exe" if args.platform == "windows" else ""
        for name in (f"cyxwiz-installer{suffix}", f"cyxwiz-backend-pack-installer{suffix}"):
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.child, destination)
            destination.chmod(0o755)
        for relative in ("runtime/trust/trusted-keys.json", "runtime/catalogs/current.json"):
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("{}\n", encoding="utf-8")

        output = root / "release"
        package_args = packager.parse_args([
            str(stage), str(output),
            "--bundle-version", "1.0.0",
            "--cyxwiz-release", "0.1.0",
            "--release-channel", "alpha",
            "--platform", args.platform,
            "--architecture", args.architecture,
            "--minimum-setup-version", "1.0.0",
            "--generated-utc", "2026-01-01T00:00:00Z",
            "--expires-utc", "2099-01-01T00:00:00Z",
        ])
        _, descriptor, _ = packager.run(package_args)
        private_key = root / "installer.pem"
        public_der = root / "installer-public.der"
        subprocess.run(
            [args.openssl, "genpkey", "-algorithm", "ED25519", "-out", private_key],
            check=True, capture_output=True,
        )
        subprocess.run(
            [args.openssl, "pkey", "-in", private_key, "-pubout", "-outform", "DER", "-out", public_der],
            check=True, capture_output=True,
        )
        if signer.main([
            str(descriptor), "--private-key", str(private_key),
            "--key-id", "setup-test-key", "--openssl", args.openssl,
        ]) != 0:
            raise RuntimeError("cannot sign setup fixture")
        encoded_key = base64.urlsafe_b64encode(public_der.read_bytes()[-32:]).decode("ascii").rstrip("=")
        trust = root / "trusted-keys.json"
        trust.write_text(json.dumps({
            "schema_version": 1,
            "keys": [{
                "key_id": "setup-test-key",
                "algorithm": "ed25519",
                "public_key": encoded_key,
                "roles": ["installer"],
                "revoked": False,
            }],
        }), encoding="utf-8")
        cache = root / "cache"
        command = [
            str(args.setup), "--descriptor", str(descriptor),
            "--trust-store", str(trust), "--cache-root", str(cache),
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            raise RuntimeError(f"setup failed ({completed.returncode}): {completed.stdout}{completed.stderr}")
        document = json.loads(descriptor.read_text(encoding="utf-8"))
        bundle_root = cache / "bundles" / document["signed"]["bundle_id"]
        if not (bundle_root / f"cyxwiz-installer{suffix}").is_file():
            raise RuntimeError("setup did not publish the verified installer")

        document["signed"]["cyxwiz_release"] = "9.9.9"
        descriptor.write_text(json.dumps(document), encoding="utf-8")
        rejected = subprocess.run(command, text=True, capture_output=True)
        if rejected.returncode == 0 or "descriptor was rejected" not in rejected.stderr:
            raise RuntimeError("setup accepted a descriptor modified after signing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
