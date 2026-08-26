#!/usr/bin/env python3
"""Validate a staged standalone-installer package on a fresh machine."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any


class PackageSmokeError(RuntimeError):
    pass


CONTRACT_TESTS = (
    "test_backend_pack_manager_model",
    "test_backend_pack_state_service",
    "test_backend_pack_installer",
    "test_backend_pack_metadata_verifier",
    "test_backend_pack_delivery",
    "test_backend_pack_maintenance",
    "test_backend_pack_maintenance_request",
    "test_backend_pack_lifecycle_service",
    "test_runtime_bootstrapper",
    "test_installer_verification_summary",
    "test_product_installation_receipt",
    "test_product_removal_authorization",
    "test_product_removal_cleanup",
    "test_product_removal_quarantine",
    "test_product_removal_finalizer",
    "test_product_removal_handoff",
    "test_product_removal_request",
    "test_product_removal_transaction",
    "test_product_registration",
)

CONTRACT_SUPPORT_EXECUTABLES = (
    "cyxwiz-product-removal-finalizer",
    "cyxwiz-runtime-bootstrapper",
    "test_product_removal_finalizer_child",
    "test_runtime_bootstrapper_child",
)


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise PackageSmokeError(f"Required package file is missing: {path.name}")
    return path


def load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(require_file(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PackageSmokeError(
            f"Required package metadata is invalid: {path.name}: {error}"
        ) from error
    if not isinstance(document, dict):
        raise PackageSmokeError(
            f"Required package metadata is not an object: {path.name}"
        )
    return document


def require_bootstrap_metadata(stage: Path) -> list[Path]:
    runtime = stage / "runtime"
    trust_path = runtime / "trust" / "trusted-keys.json"
    catalog_path = runtime / "catalogs" / "current.json"
    trust = load_json(trust_path)
    catalog = load_json(catalog_path)
    if trust.get("schema_version") != 1 or not isinstance(trust.get("keys"), list):
        raise PackageSmokeError("The packaged trust store violates schema 1")
    if catalog.get("schema_version") != 1 or catalog.get("kind") != (
        "cyxwiz-backend-pack-catalog"
    ):
        raise PackageSmokeError("The packaged backend-pack catalog violates schema 1")
    signed = catalog.get("signed")
    packs = signed.get("packs") if isinstance(signed, dict) else None
    if not isinstance(packs, list) or not packs:
        raise PackageSmokeError("The packaged backend-pack catalog contains no packs")

    manifest_directory = runtime / "catalogs" / "manifests"
    manifests = sorted(manifest_directory.glob("*.json"))
    manifest_backends: dict[str, str] = {}
    for path in manifests:
        document = load_json(path)
        body = document.get("signed")
        if (
            document.get("schema_version") != 1
            or document.get("kind") != "cyxwiz-backend-pack-manifest"
            or not isinstance(body, dict)
            or not isinstance(body.get("pack_id"), str)
            or not isinstance(body.get("backend"), str)
        ):
            raise PackageSmokeError(
                f"The packaged backend-pack manifest violates schema 1: {path.name}"
            )
        manifest_backends[body["pack_id"]] = body["backend"]

    catalog_ids = {
        pack.get("pack_id") for pack in packs
        if isinstance(pack, dict) and isinstance(pack.get("pack_id"), str)
    }
    if catalog_ids != set(manifest_backends):
        raise PackageSmokeError(
            "The packaged catalog and cached manifest identities differ"
        )
    if "cpu" not in manifest_backends.values() or not any(
        backend != "cpu" for backend in manifest_backends.values()
    ):
        raise PackageSmokeError(
            "The bootstrap metadata must provide a CPU base and an optional backend pack"
        )
    return [trust_path, catalog_path, *manifests]


def runtime_environment(
    runtime_root: Path, executable_directory: Path
) -> dict[str, str]:
    environment = os.environ.copy()
    if os.name == "nt":
        windows = Path(environment.get("SystemRoot", r"C:\Windows"))
        environment["PATH"] = os.pathsep.join(
            (
                str(executable_directory),
                str(runtime_root),
                str(windows / "System32"),
                str(windows),
            )
        )
    elif platform.system() == "Darwin":
        environment["DYLD_LIBRARY_PATH"] = os.pathsep.join(
            (str(runtime_root), str(runtime_root / "Frameworks"))
        )
    else:
        environment["LD_LIBRARY_PATH"] = str(runtime_root)
    return environment


def run_checked(
    path: Path,
    arguments: list[str],
    expected: int,
    runtime_root: Path,
) -> dict[str, Any]:
    environment = runtime_environment(runtime_root, path.parent)
    started = time.perf_counter_ns()
    result = subprocess.run(
        [str(path), *arguments],
        cwd=path.parent,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    duration_ms = round((time.perf_counter_ns() - started) / 1_000_000, 3)
    if result.returncode != expected:
        output = result.stdout.strip()
        raise PackageSmokeError(
            f"{path.name} returned {result.returncode}, expected {expected}: {output}"
        )
    return {
        "name": path.name,
        "arguments": arguments,
        "expected_exit_code": expected,
        "observed_exit_code": result.returncode,
        "duration_ms": duration_ms,
        "status": "passed",
    }


def dependency_output(command: list[str], environment: dict[str, str]) -> str:
    result = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )
    if result.returncode != 0:
        raise PackageSmokeError(
            f"Dependency inspection failed ({' '.join(command)}): {result.stdout.strip()}"
        )
    return result.stdout


def audit_dependencies(
    executables: list[Path], runtime_root: Path
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    system = platform.system()
    forbidden = ("vcpkg_installed", "/home/runner/work/", "/Users/runner/work/")
    for executable in executables:
        if system == "Linux":
            output = dependency_output(
                ["ldd", str(executable)],
                runtime_environment(runtime_root, executable.parent),
            )
            if "not found" in output:
                raise PackageSmokeError(
                    f"{executable.name} has an unresolved dependency:\n{output}"
                )
        elif system == "Darwin":
            output = dependency_output(
                ["otool", "-L", str(executable)],
                runtime_environment(runtime_root, executable.parent),
            )
            output = "\n".join(output.splitlines()[1:])
        else:
            continue
        if any(marker in output for marker in forbidden):
            raise PackageSmokeError(
                f"{executable.name} retains a build-machine dependency:\n{output}"
            )
    return {
        "status": "passed",
        "duration_ms": round((time.perf_counter_ns() - started) / 1_000_000, 3),
        "executables": [path.name for path in executables],
        "forbidden_build_paths": list(forbidden),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_inventory(stage: Path) -> tuple[list[dict[str, Any]], int, int]:
    files = []
    installed_size = 0
    validation_size = 0
    for path in sorted(item for item in stage.rglob("*") if item.is_file()):
        size = path.stat().st_size
        relative_path = path.relative_to(stage)
        role = "validation" if relative_path.parts[0] == "smoke" else "install_payload"
        if role == "validation":
            validation_size += size
        else:
            installed_size += size
        files.append({
            "path": relative_path.as_posix(),
            "role": role,
            "size_bytes": size,
            "sha256": sha256_file(path),
        })
    return files, installed_size, validation_size


def hardware_matrix() -> list[dict[str, str]]:
    reason = "physical_supported_hardware_and_provider_required"
    return [
        {"route": "cuda", "status": "not_run", "reason": reason},
        {"route": "opencl", "status": "not_run", "reason": reason},
        {"route": "oneapi", "status": "not_run", "reason": reason},
    ]


def build_evidence(
    artifact_id: str,
    files: list[dict[str, Any]],
    installed_size: int,
    validation_size: int,
    dependency_audit: dict[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "artifact_kind": "standalone_installer",
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "installed_size_bytes": installed_size,
        "validation_payload_size_bytes": validation_size,
        "file_count": len(files),
        "files": files,
        "dependency_audit": dependency_audit,
        "checks": checks,
        "accelerator_routes": hardware_matrix(),
        "result": "passed",
    }


def verify(stage: Path, artifact_id: str) -> dict[str, Any]:
    stage = stage.resolve()
    suffix = ".exe" if os.name == "nt" else ""
    require_bootstrap_metadata(stage)
    installer = require_file(stage / f"cyxwiz-installer{suffix}")
    helper = require_file(stage / f"cyxwiz-backend-pack-installer{suffix}")
    contract_tests = [
        require_file(stage / "smoke" / f"{name}{suffix}")
        for name in CONTRACT_TESTS
    ]
    contract_support = [
        require_file(stage / "smoke" / f"{name}{suffix}")
        for name in CONTRACT_SUPPORT_EXECUTABLES
    ]
    validation_executables = [*contract_tests, *contract_support]
    if os.name != "nt":
        for executable in (installer, helper, *validation_executables):
            executable.chmod(executable.stat().st_mode | 0o111)

    dependency_audit = audit_dependencies(
        [installer, helper, *validation_executables], stage
    )
    checks = [
        run_checked(installer, ["--package-smoke"], 0, stage),
        run_checked(helper, [], 78, stage),
    ]
    checks.extend(run_checked(test, [], 0, stage) for test in contract_tests)
    files, installed_size, validation_size = package_inventory(stage)
    evidence = build_evidence(
        artifact_id,
        files,
        installed_size,
        validation_size,
        dependency_audit,
        checks,
    )
    print(f"Installer package smoke passed: {stage}")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=Path)
    parser.add_argument("--artifact-id", default="local-installer-stage")
    parser.add_argument("--evidence", type=Path)
    arguments = parser.parse_args()
    try:
        evidence = verify(arguments.stage, arguments.artifact_id)
        if arguments.evidence:
            arguments.evidence.parent.mkdir(parents=True, exist_ok=True)
            arguments.evidence.write_text(
                json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
            )
            print(f"Clean-machine evidence written: {arguments.evidence}")
    except (OSError, subprocess.SubprocessError, PackageSmokeError) as error:
        print(f"Installer package smoke failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
