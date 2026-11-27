from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unicodedata
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


# A plugin pack's name and description: the cases this suite and the
# bootstrapper's verifier test both run.
DISPLAY_TEXT_CASES = Path(__file__).resolve().parent / "fixtures" / "plugin_display_text_cases.json"

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


def plugin_manifest() -> dict:
    pack_id = "cyxwiz-plugin-mujoco-4.0.1-1-win64"
    return {
        "schema_version": 2,
        "kind": "cyxwiz-backend-pack-manifest",
        "signed": {
            "pack_id": pack_id,
            "pack_kind": "plugin_pack",
            "plugin_id": "com.cyxwiz.simulation.mujoco",
            "package_version": "1.0.0",
            "platform": "win64",
            "architecture": "x86_64",
            "cyxwiz_release": {"minimum": "0.2.0", "maximum": "0.2.x"},
            "companion_base_id": "cyxwiz-base-0.2.0-win64",
            "conflicts": [],
            "support_status": "supported",
            "components": [
                {
                    "path": "plugin.json",
                    "size": 512,
                    "sha256": SHA,
                    "source": "cyxwiz-plugin",
                    "executable": False,
                },
                {
                    "path": "bin/mujoco.dll",
                    "size": 1234,
                    "sha256": SHA,
                    "source": "mujoco",
                    "executable": True,
                },
                {
                    "path": "THIRD_PARTY_LICENSES/MuJoCo/LICENSE.txt",
                    "size": 42,
                    "sha256": SHA,
                    "source": "mujoco-license",
                    "executable": False,
                },
            ],
            "licenses": [
                {
                    "component": "mujoco",
                    "path": "THIRD_PARTY_LICENSES/MuJoCo/LICENSE.txt",
                }
            ],
            "archive": {
                "file_name": f"{pack_id}.zip",
                "size": 4096,
                "sha256": SHA,
            },
            "generated_utc": "2026-11-06T12:00:00Z",
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

    def test_plugin_pack_manifest_passes(self) -> None:
        contract.validate_pack_manifest(plugin_manifest())
        contract.validate_plugin_pack_manifest(plugin_manifest())

    def test_plugin_pack_carries_the_plugins_own_name_and_description(self) -> None:
        # A pack made before the manifest carried them (plugin_manifest()) passes
        # above; each of the two is optional on its own.
        document = plugin_manifest()
        document["signed"]["plugin_id"] = "org.example.fluid-lab"
        document["signed"]["plugin_name"] = "Fluid Lab"
        contract.validate_pack_manifest(document)
        document["signed"]["plugin_description"] = "Fluid simulation: tanks, pipes and pumps (\u00e9tude)."
        contract.validate_pack_manifest(document)

    def test_plugin_pack_rejects_unusable_names_and_descriptions(self) -> None:
        for field, value, message in (
            ("plugin_name", "", "non-empty string"),
            ("plugin_name", 7, "non-empty string"),
            ("plugin_name", "x" * (contract.PLUGIN_NAME_MAX_BYTES + 1), "longer than"),
            ("plugin_name", "\u00e9" * contract.PLUGIN_NAME_MAX_BYTES, "longer than"),
            ("plugin_name", "Fluid\nLab", "control character"),
            ("plugin_description", None, "non-empty string"),
            ("plugin_description", "x" * (contract.PLUGIN_DESCRIPTION_MAX_BYTES + 1), "longer than"),
            ("plugin_description", "tab\there", "control character"),
        ):
            document = plugin_manifest()
            document["signed"][field] = value
            with self.assertRaisesRegex(contract.ContractError, message):
                contract.validate_pack_manifest(document)
        # Still strict: another new field is refused
        document = plugin_manifest()
        document["signed"]["plugin_author"] = "Example"
        with self.assertRaisesRegex(contract.ContractError, "unsupported fields"):
            contract.validate_pack_manifest(document)

    def test_plugin_display_text_cases_shared_with_the_verifier(self) -> None:
        # The same file is run by test_backend_pack_metadata_verifier.cpp:
        # packaging and the bootstrapper judge every case alike.
        cases = json.loads(DISPLAY_TEXT_CASES.read_text(encoding="ascii"))["cases"]
        self.assertGreater(len(cases), 30)
        for case in cases:
            text = case["text"] * case.get("repeat", 1) + case.get("suffix", "")
            with self.subTest(case["why"]):
                if case["accepted"]:
                    self.assertEqual(text, contract.display_text(text, "name", case["maximum_bytes"]))
                else:
                    with self.assertRaises(contract.ContractError):
                        contract.display_text(text, "name", case["maximum_bytes"])

    def test_plugin_display_text_limits_are_bytes_of_utf8(self) -> None:
        name = "\u00e9" * (contract.PLUGIN_NAME_MAX_BYTES // 2)
        self.assertEqual(contract.PLUGIN_NAME_MAX_BYTES, len(name.encode("utf-8")))
        description = "\U0001F30A" * (contract.PLUGIN_DESCRIPTION_MAX_BYTES // 4)
        self.assertEqual(contract.PLUGIN_DESCRIPTION_MAX_BYTES, len(description.encode("utf-8")))
        document = plugin_manifest()
        document["signed"]["plugin_name"] = name
        document["signed"]["plugin_description"] = description
        contract.validate_pack_manifest(document)
        document["signed"]["plugin_name"] = name + "x"
        with self.assertRaisesRegex(contract.ContractError, "longer than 128 bytes"):
            contract.validate_pack_manifest(document)
        document["signed"]["plugin_name"] = name
        document["signed"]["plugin_description"] = description + "x"
        with self.assertRaisesRegex(contract.ContractError, "longer than 512 bytes"):
            contract.validate_pack_manifest(document)

    def test_plugin_display_text_refuses_a_lone_surrogate_cleanly(self) -> None:
        # No UTF-8 form: a ContractError, never a UnicodeEncodeError
        for text in ("Fluid\ud800Lab", "\udfff", "Fluid \udc00"):
            with self.assertRaisesRegex(contract.ContractError, "not valid Unicode text"):
                contract.display_text(text, "plugin.json name", contract.PLUGIN_NAME_MAX_BYTES)
        document = plugin_manifest()
        document["signed"]["plugin_name"] = "Fluid\ud800Lab"
        with self.assertRaisesRegex(contract.ContractError, "not valid Unicode text"):
            contract.validate_pack_manifest(document)
        with self.assertRaisesRegex(contract.ContractError, "not valid Unicode text"):
            contract.canonical_json_bytes(document["signed"])

    def test_plugin_display_text_names_what_is_wrong(self) -> None:
        for text, message in (
            ("   ", "is blank"),
            ("\u00a0\u3000", "is blank"),
            ("Fluid\u0085Lab", "control character"),
            ("Fluid\u202eLab", "invisible or direction-changing"),
            ("Fluid\u200bLab", "invisible or direction-changing"),
        ):
            with self.assertRaisesRegex(contract.ContractError, message):
                contract.display_text(text, "name", contract.PLUGIN_NAME_MAX_BYTES)

    def test_plugin_display_text_list_covers_this_pythons_unicode(self) -> None:
        # The rules are a fixed list (the verifier has no Unicode database).
        # Every control, format character, line or paragraph separator and
        # surrogate this Python knows must be refused by it, and every space
        # must count as blank: a newer Unicode that adds one fails here.
        for code_point in range(0x110000):
            category = unicodedata.category(chr(code_point))
            if category in ("Cc", "Cf", "Cs", "Zl", "Zp"):
                with self.assertRaises(contract.ContractError, msg=hex(code_point)):
                    contract.display_text("a" + chr(code_point), "name", 128)
            elif category == "Zs":
                with self.assertRaisesRegex(contract.ContractError, "is blank", msg=hex(code_point)):
                    contract.display_text(chr(code_point), "name", 128)

    def test_plugin_pack_rejects_arrayfire_and_device_fields(self) -> None:
        for name, value in (
            ("arrayfire", {"version": "3.10.0", "abi": "arrayfire-3.10"}),
            ("backend", "cpu"),
            ("compatibility", {}),
            ("runtime_set_id", "arrayfire-3.10.0-win64-v1"),
        ):
            document = plugin_manifest()
            document["signed"][name] = value
            with self.assertRaisesRegex(contract.ContractError, "unsupported fields"):
                contract.validate_pack_manifest(document)

    def test_schema_1_manifest_cannot_be_a_plugin_pack(self) -> None:
        document = pack_manifest()
        document["signed"]["pack_kind"] = "plugin_pack"
        with self.assertRaisesRegex(contract.ContractError, "pack_kind"):
            contract.validate_pack_manifest(document)

    def test_schema_2_manifest_must_be_a_plugin_pack(self) -> None:
        document = plugin_manifest()
        document["signed"]["pack_kind"] = "backend_pack"
        with self.assertRaisesRegex(contract.ContractError, "plugin_pack"):
            contract.validate_pack_manifest(document)

    def test_plugin_pack_requires_a_companion_base(self) -> None:
        document = plugin_manifest()
        document["signed"]["companion_base_id"] = None
        with self.assertRaisesRegex(contract.ContractError, "companion_base_id"):
            contract.validate_pack_manifest(document)

    def test_unknown_schema_version_fails_closed(self) -> None:
        document = plugin_manifest()
        document["schema_version"] = 3
        with self.assertRaisesRegex(contract.ContractError, "schema_version"):
            contract.validate_pack_manifest(document)

    def test_support_status_is_read_from_either_schema(self) -> None:
        backend = pack_manifest()["signed"]
        plugin = plugin_manifest()["signed"]
        plugin["support_status"] = "diagnostic"
        self.assertEqual(contract.manifest_support_status(backend), "supported")
        self.assertEqual(contract.manifest_support_status(plugin), "diagnostic")

    def test_active_plugins_passes_and_rejects_duplicates(self) -> None:
        document = {
            "schema_version": 1,
            "base_pack_id": "cyxwiz-base-0.2.0-win64",
            "plugins": [
                {
                    "plugin_id": "com.cyxwiz.simulation.mujoco",
                    "pack_id": "cyxwiz-plugin-mujoco-4.0.1-1-win64",
                }
            ],
        }
        contract.validate_active_plugins(document)
        document["plugins"].append(dict(document["plugins"][0], pack_id="other-v2"))
        with self.assertRaisesRegex(contract.ContractError, "duplicate active plugin"):
            contract.validate_active_plugins(document)

    def test_active_plugins_rejects_runtime_identity_fields(self) -> None:
        document = {
            "schema_version": 1,
            "base_pack_id": "cyxwiz-base-0.2.0-win64",
            "plugins": [],
            "generation": 3,
        }
        with self.assertRaisesRegex(contract.ContractError, "unsupported fields"):
            contract.validate_active_plugins(document)

    def test_plugin_composition_accepts_its_base(self) -> None:
        contract.validate_plugin_composition(pack_manifest("base", "cpu"), plugin_manifest())

    def test_plugin_composition_rejects_another_base_or_platform(self) -> None:
        plugin = plugin_manifest()
        plugin["signed"]["companion_base_id"] = "cyxwiz-base-0.3.0-win64"
        with self.assertRaisesRegex(contract.ContractError, "different companion base"):
            contract.validate_plugin_composition(pack_manifest("base", "cpu"), plugin)
        plugin = plugin_manifest()
        plugin["signed"]["platform"] = "linux64"
        with self.assertRaisesRegex(contract.ContractError, "different platforms"):
            contract.validate_plugin_composition(pack_manifest("base", "cpu"), plugin)

    def test_backend_composition_rejects_a_plugin_pack(self) -> None:
        with self.assertRaises(contract.ContractError):
            contract.validate_runtime_composition(pack_manifest("base", "cpu"), plugin_manifest())

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
