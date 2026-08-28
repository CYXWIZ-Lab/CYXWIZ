from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "backend_pack_contract.py"
)
SPEC = importlib.util.spec_from_file_location("backend_pack_contract", SCRIPT)
assert SPEC and SPEC.loader
contract = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = contract
SPEC.loader.exec_module(contract)


SHA = "a" * 64
SIGNATURE = "A" * 86


def signature() -> dict:
    return {
        "key_id": "release-2026",
        "algorithm": "ed25519",
        "value": SIGNATURE,
    }


def pack_manifest(kind: str = "backend_pack", backend: str = "opencl") -> dict:
    pack_id = "cyxwiz-af-opencl-3.10.0-1-win64"
    companion = "cyxwiz-base-0.2.0-win64"
    if kind == "base":
        backend = "cpu"
        pack_id = companion
        companion = None
    return {
        "schema_version": 1,
        "kind": "cyxwiz-backend-pack-manifest",
        "signed": {
            "pack_id": pack_id,
            "pack_kind": kind,
            "backend": backend,
            "package_version": "1.0.0",
            "platform": "win64",
            "architecture": "x86_64",
            "runtime_set_id": "arrayfire-3.10.0-win64-v1",
            "cyxwiz_release": {"minimum": "0.2.0", "maximum": "0.2.x"},
            "arrayfire": {"version": "3.10.0", "abi": "arrayfire-3.10"},
            "companion_base_id": companion,
            "conflicts": [],
            "compatibility": {
                "device_kinds": ["gpu"],
                "cpu_features": [],
                "provider_types": ["opencl-icd"],
                "minimum_driver_versions": {"intel": "31.0.101.2115"},
                "tested_driver_ranges": {"intel": ">=31.0.101.2115"},
                "minimum_identity_confidence": "stable_hardware",
                "recommendation_targets": ["oneapi", "cpu"],
                "operation_matrix_id": "cyxwiz-route-qualification-v1",
                "training_scope": ["dense", "loss", "optimizer"],
                "support_status": "supported",
            },
            "components": [
                {
                    "path": "runtime/afopencl.dll",
                    "size": 1234,
                    "sha256": SHA,
                    "source": "arrayfire",
                    "executable": True,
                },
                {
                    "path": "THIRD_PARTY_LICENSES/ArrayFire/LICENSE.txt",
                    "size": 42,
                    "sha256": SHA,
                    "source": "arrayfire-license",
                    "executable": False,
                },
            ],
            "licenses": [
                {
                    "component": "arrayfire",
                    "path": "THIRD_PARTY_LICENSES/ArrayFire/LICENSE.txt",
                }
            ],
            "archive": {
                "file_name": f"{pack_id}.zip",
                "size": 4096,
                "sha256": SHA,
            },
            "generated_utc": "2026-08-13T20:00:00Z",
        },
        "signatures": [signature()],
    }


def catalog() -> dict:
    return {
        "schema_version": 1,
        "kind": "cyxwiz-backend-pack-catalog",
        "signed": {
            "catalog_id": "production-2026-08",
            "generated_utc": "2026-08-13T20:00:00Z",
            "expires_utc": "2026-09-13T20:00:00Z",
            "minimum_client_version": "0.2.0",
            "packs": [
                {
                    "pack_id": "cyxwiz-af-opencl-3.10.0-1-win64",
                    "manifest_url": "https://downloads.cyxwiz.com/packs/opencl.json",
                    "manifest_sha256": SHA,
                    "signing_key_id": "release-2026",
                    "support_status": "supported",
                }
            ],
        },
        "signatures": [signature()],
    }


def trust_root() -> dict:
    return {
        "schema_version": 1,
        "keys": [
            {
                "key_id": "release-2026",
                "algorithm": "ed25519",
                "public_key": "A" * 43,
                "roles": ["catalog", "pack"],
                "revoked": False,
            }
        ],
    }


class BackendPackContractTests(unittest.TestCase):
    def test_optional_pack_manifest_passes(self) -> None:
        contract.validate_pack_manifest(pack_manifest())

    def test_cpu_base_manifest_passes(self) -> None:
        document = pack_manifest("base", "cpu")
        document["signed"]["compatibility"]["device_kinds"] = ["cpu"]
        document["signed"]["compatibility"]["provider_types"] = ["arrayfire-cpu"]
        contract.validate_pack_manifest(document)

    def test_optional_cpu_pack_is_rejected(self) -> None:
        with self.assertRaisesRegex(contract.ContractError, "base pack must be CPU"):
            contract.validate_pack_manifest(pack_manifest("backend_pack", "cpu"))

    def test_path_traversal_is_rejected_before_extraction(self) -> None:
        document = pack_manifest()
        document["signed"]["components"][0]["path"] = "../afopencl.dll"
        with self.assertRaisesRegex(contract.ContractError, "unsafe path"):
            contract.validate_pack_manifest(document)

    def test_noncanonical_repeated_path_separator_is_rejected(self) -> None:
        document = pack_manifest()
        document["signed"]["components"][0]["path"] = "runtime//afopencl.dll"
        with self.assertRaisesRegex(contract.ContractError, "unsafe path"):
            contract.validate_pack_manifest(document)

    def test_case_insensitive_duplicate_paths_are_rejected(self) -> None:
        document = pack_manifest()
        duplicate = copy.deepcopy(document["signed"]["components"][0])
        duplicate["path"] = "runtime/AFOPENCL.dll"
        document["signed"]["components"].append(duplicate)
        with self.assertRaisesRegex(contract.ContractError, "duplicate canonical"):
            contract.validate_pack_manifest(document)

    def test_license_must_reference_packaged_component(self) -> None:
        document = pack_manifest()
        document["signed"]["licenses"][0]["path"] = "LICENSE-missing.txt"
        with self.assertRaisesRegex(contract.ContractError, "not a packaged component"):
            contract.validate_pack_manifest(document)

    def test_unknown_manifest_field_fails_closed(self) -> None:
        document = pack_manifest()
        document["signed"]["post_install_script"] = "setup.cmd"
        with self.assertRaisesRegex(contract.ContractError, "unsupported fields"):
            contract.validate_pack_manifest(document)

    def test_unknown_recommendation_backend_is_rejected(self) -> None:
        document = pack_manifest()
        document["signed"]["compatibility"]["recommendation_targets"] = ["vulkan"]
        with self.assertRaisesRegex(contract.ContractError, "unknown backend"):
            contract.validate_pack_manifest(document)

    def test_manifest_requires_ed25519_signature(self) -> None:
        document = pack_manifest()
        document["signatures"][0]["algorithm"] = "rsa-sha256"
        with self.assertRaisesRegex(contract.ContractError, "must be ed25519"):
            contract.validate_pack_manifest(document)

    def test_catalog_passes(self) -> None:
        contract.validate_catalog(catalog())

    def test_catalog_requires_https_manifest_url(self) -> None:
        document = catalog()
        document["signed"]["packs"][0]["manifest_url"] = "http://example/p.json"
        with self.assertRaisesRegex(contract.ContractError, "must use HTTPS"):
            contract.validate_catalog(document)

    def test_catalog_expiry_must_follow_generation(self) -> None:
        document = catalog()
        document["signed"]["expires_utc"] = "2026-08-12T20:00:00Z"
        with self.assertRaisesRegex(contract.ContractError, "later than"):
            contract.validate_catalog(document)

    def test_trust_root_passes(self) -> None:
        contract.validate_trust_root(trust_root())

    def test_trust_root_accepts_separate_installer_authority(self) -> None:
        document = trust_root()
        document["keys"][0]["roles"] = ["installer"]
        contract.validate_trust_root(document)

    def test_trust_root_rejects_duplicate_roles(self) -> None:
        document = trust_root()
        document["keys"][0]["roles"] = ["pack", "pack"]
        with self.assertRaisesRegex(contract.ContractError, "must be unique"):
            contract.validate_trust_root(document)

    def test_trust_root_rejects_unknown_role(self) -> None:
        document = trust_root()
        document["keys"][0]["roles"] = ["publisher"]
        with self.assertRaisesRegex(contract.ContractError, "unsupported role"):
            contract.validate_trust_root(document)

    def test_trust_root_rejects_unknown_fields(self) -> None:
        document = trust_root()
        document["keys"][0]["private_key"] = "forbidden"
        with self.assertRaisesRegex(contract.ContractError, "unsupported fields"):
            contract.validate_trust_root(document)

    def test_active_runtime_rejects_duplicate_backend(self) -> None:
        document = {
            "schema_version": 1,
            "runtime_set_id": "arrayfire-3.10.0-win64-v1",
            "generation": 2,
            "base_pack_id": "cyxwiz-base-0.2.0-win64",
            "packs": [
                {"backend": "opencl", "pack_id": "opencl-v1"},
                {"backend": "opencl", "pack_id": "opencl-v2"},
            ],
        }
        with self.assertRaisesRegex(contract.ContractError, "duplicate active backend"):
            contract.validate_active_runtime(document)

    def test_runtime_composition_accepts_matching_base_and_pack(self) -> None:
        base = pack_manifest("base", "cpu")
        pack = pack_manifest()
        base["signed"]["runtime_set_id"] = pack["signed"]["runtime_set_id"]
        base["signed"]["arrayfire"] = copy.deepcopy(pack["signed"]["arrayfire"])
        contract.validate_runtime_composition(base, pack)

    def test_runtime_composition_rejects_arrayfire_abi_mixing(self) -> None:
        base = pack_manifest("base", "cpu")
        pack = pack_manifest()
        base["signed"]["runtime_set_id"] = pack["signed"]["runtime_set_id"]
        pack["signed"]["arrayfire"]["abi"] = "arrayfire-3.9"
        with self.assertRaisesRegex(contract.ContractError, "incompatible ArrayFire ABI"):
            contract.validate_runtime_composition(base, pack)

    def test_canonical_json_is_key_order_independent(self) -> None:
        left = {"b": [2, 1], "a": {"d": "x", "c": True}}
        right = {"a": {"c": True, "d": "x"}, "b": [2, 1]}
        self.assertEqual(
            contract.canonical_json_bytes(left),
            contract.canonical_json_bytes(right),
        )

    def test_canonical_json_rejects_float_values(self) -> None:
        with self.assertRaisesRegex(contract.ContractError, "floating-point"):
            contract.canonical_json_bytes({"size": 1.5})


if __name__ == "__main__":
    unittest.main()
