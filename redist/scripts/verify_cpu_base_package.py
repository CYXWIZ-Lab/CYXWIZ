#!/usr/bin/env python3
"""Install and exercise a native CPU-base artifact on a fresh runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any
import zipfile

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backend_pack_contract import (  # noqa: E402
    ContractError,
    canonical_json_bytes,
    validate_active_runtime,
    validate_pack_manifest,
)
from backend_pack_target import (  # noqa: E402
    BackendPackTargetError,
    detect_backend_pack_target,
)
from macho_runtime_closure import (  # noqa: E402
    is_macho,
    is_system_dependency,
    parse_otool_dependencies,
)


class CpuBaseSmokeError(RuntimeError):
    pass


OPERATIONS = (
    "constant",
    "tensor_row_major",
    "cyxwiz_bce_forward",
    "cyxwiz_bce_backward",
    "dense_compute_benchmark",
)
OVERRIDE_VARIABLES = (
    "AF_PATH",
    "AF_PLUGIN_PATH",
    "CYXWIZ_ARRAYFIRE_DIR",
    "AF_BUILD_PATH",
    "AF_BUILD_LIB_CUSTOM_PATH",
    "PYTHONHOME",
    "PYTHONPATH",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path, archive: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    validation_document = document
    if not document.get("signatures"):
        validation_document = {
            **document,
            "signatures": [{
                "key_id": "ci-structural-validation",
                "algorithm": "ed25519",
                "value": "A" * 86,
            }],
        }
    validate_pack_manifest(validation_document)
    signed = document["signed"]
    signature_input = archive.with_suffix(archive.suffix + ".signed.json")
    if not signature_input.is_file():
        raise CpuBaseSmokeError("Canonical signature input is missing")
    if signature_input.read_bytes() != canonical_json_bytes(signed):
        raise CpuBaseSmokeError("Canonical signature input differs from the manifest")
    if signed["pack_kind"] != "base" or signed["backend"] != "cpu":
        raise CpuBaseSmokeError("Manifest does not describe a CPU base")
    try:
        target = detect_backend_pack_target()
    except BackendPackTargetError as error:
        raise CpuBaseSmokeError(str(error)) from error
    if (
        signed["platform"] != target.platform
        or signed["architecture"] != target.architecture
    ):
        raise CpuBaseSmokeError(
            "Manifest target differs from the clean-runner host: "
            f"{signed['platform']}-{signed['architecture']}"
        )
    archive_contract = signed["archive"]
    if archive.name != archive_contract["file_name"]:
        raise CpuBaseSmokeError("Archive name differs from its manifest")
    if archive.stat().st_size != archive_contract["size"]:
        raise CpuBaseSmokeError("Archive size differs from its manifest")
    if sha256_file(archive) != archive_contract["sha256"]:
        raise CpuBaseSmokeError("Archive digest differs from its manifest")
    return document


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise CpuBaseSmokeError(
                    f"Archive member escapes the base directory: {member.filename}"
                )
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            mode = (member.external_attr >> 16) & 0o777
            if mode:
                target.chmod(mode)


def install(archive: Path, manifest: dict[str, Any], install_root: Path) -> Path:
    if install_root.exists() and any(install_root.iterdir()):
        raise CpuBaseSmokeError(f"Install root is not empty: {install_root}")
    install_root.mkdir(parents=True, exist_ok=True)
    signed = manifest["signed"]
    runtime_root = install_root / "runtime"
    base = runtime_root / "base" / signed["pack_id"]
    safe_extract(archive, base)
    suffix = ".exe" if os.name == "nt" else ""
    bootstrapper = base / f"cyxwiz-runtime-bootstrapper{suffix}"
    if not bootstrapper.is_file():
        raise CpuBaseSmokeError("CPU base is missing its runtime bootstrapper")
    shutil.copy2(bootstrapper, install_root / bootstrapper.name)
    finalizer = base / f"cyxwiz-product-removal-finalizer{suffix}"
    if not finalizer.is_file():
        raise CpuBaseSmokeError("CPU base is missing its product-removal finalizer")
    shutil.copy2(finalizer, install_root / finalizer.name)
    state = {
        "schema_version": 1,
        "runtime_set_id": signed["runtime_set_id"],
        "generation": 1,
        "base_pack_id": signed["pack_id"],
        "packs": [],
    }
    validate_active_runtime(state)
    (runtime_root / "active-runtime.json").write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )
    return base


def contaminated_environment(install_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    marker = install_root / "cyxwiz-untrusted-marker"
    for name in OVERRIDE_VARIABLES:
        environment[name] = str(marker)
    if os.name == "nt":
        windows = Path(environment.get("SystemRoot", r"C:\Windows"))
        environment["PATH"] = os.pathsep.join(
            (str(marker), str(windows / "System32"))
        )
    else:
        environment["PATH"] = os.pathsep.join((str(marker), "/usr/bin", "/bin"))
    return environment


def package_environment(install_root: Path, base: Path) -> dict[str, str]:
    environment = contaminated_environment(install_root)
    for name in OVERRIDE_VARIABLES:
        environment.pop(name, None)
    library_directories = (
        base,
        base / "lib",
        base / "arrayfire" / ("bin" if os.name == "nt" else "lib"),
        base / "python",
    )
    if os.name == "nt":
        windows = Path(environment.get("SystemRoot", r"C:\Windows"))
        environment["PATH"] = os.pathsep.join(
            (*map(str, library_directories), str(windows / "System32"))
        )
    else:
        environment["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        loader = "DYLD_LIBRARY_PATH" if platform.system() == "Darwin" else "LD_LIBRARY_PATH"
        environment[loader] = os.pathsep.join(map(str, library_directories))
    environment["CYXWIZ_ACTIVE_RUNTIME_ROOT"] = str(install_root / "runtime")
    return environment


def run_checked(
    executable: Path,
    arguments: list[str],
    environment: dict[str, str],
    timeout: int,
) -> tuple[dict[str, Any], str]:
    started = time.perf_counter_ns()
    result = subprocess.run(
        [str(executable), *arguments],
        cwd=executable.parent,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=timeout,
    )
    duration_ms = round((time.perf_counter_ns() - started) / 1_000_000, 3)
    if result.returncode != 0:
        raise CpuBaseSmokeError(
            f"{executable.name} returned {result.returncode}: {result.stdout.strip()}"
        )
    return ({
        "name": executable.name,
        "arguments": arguments,
        "observed_exit_code": result.returncode,
        "duration_ms": duration_ms,
        "output_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
        "status": "passed",
    }, result.stdout)


def parse_engine_smoke(output: str) -> None:
    expected = (
        "package_smoke schema=1 status=pass effective_backend=cpu "
        "effective_device=0 runtime_isolation=pass checksum=70"
    )
    if expected not in output:
        raise CpuBaseSmokeError("Engine did not report a passing CPU package smoke")


def parse_probe(output: str, operation: str) -> None:
    result = re.search(
        rf"probe_result schema=1 backend=cpu device_id=0 "
        rf"operation={re.escape(operation)} status=pass "
        rf"effective_backend=\d+ effective_device=0",
        output,
    )
    if result is None:
        raise CpuBaseSmokeError(f"CPU route probe did not pass {operation}")


def audit_runtime_modules(output: str, base: Path) -> list[str]:
    modules = re.findall(
        r"runtime_module point=\w+ name=(\S+) path='([^']+)'", output
    )
    unique: list[str] = []
    canonical_base = base.resolve()
    for reported_name, raw in modules:
        path = Path(raw).resolve()
        if canonical_base not in path.parents:
            name = reported_name.lower()
            if name.startswith("af") or name.startswith("mkl_rt"):
                raise CpuBaseSmokeError(
                    f"Runtime module loaded outside CPU base: {path.name}"
                )
            continue
        relative = path.relative_to(canonical_base).as_posix()
        if relative not in unique:
            unique.append(relative)
    lowered = [Path(item).name.lower() for item in unique]
    for required in ("af.dll", "afcpu.dll"):
        if required not in lowered:
            raise CpuBaseSmokeError(f"Package-local {required} was not observed")
    if not any(name.startswith("mkl_rt") for name in lowered):
        raise CpuBaseSmokeError("Package-local oneMKL CPU runtime was not observed")
    return unique


def audit_macos_dependencies(base: Path) -> list[str]:
    required = (
        base / "arrayfire" / "lib" / "libaf.dylib",
        base / "arrayfire" / "lib" / "libaf.3.dylib",
        base / "arrayfire" / "lib" / "libafcpu.dylib",
        base / "arrayfire" / "lib" / "libafcpu.3.dylib",
    )
    for path in required:
        if not path.is_file():
            raise CpuBaseSmokeError(f"Package-local {path.name} is missing")
    audited: list[str] = []
    candidates = [
        path for path in base.rglob("*")
        if path.is_file() and is_macho(path)
    ]
    for path in candidates:
        result = subprocess.run(
            ["otool", "-L", str(path)],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            raise CpuBaseSmokeError(
                f"Cannot inspect packaged Mach-O file {path.name}: {result.stdout.strip()}"
            )
        dependencies = parse_otool_dependencies(result.stdout)
        for dependency in dependencies:
            if (
                is_system_dependency(dependency)
                or dependency.startswith(("@loader_path/", "@rpath/"))
            ):
                continue
            raise CpuBaseSmokeError(
                f"{path.name} retains a non-system absolute dependency: "
                f"{dependency}"
            )
        audited.append(path.relative_to(base).as_posix())
    return audited


def package_inventory(install_root: Path) -> tuple[list[dict[str, Any]], int]:
    files: list[dict[str, Any]] = []
    total = 0
    for path in sorted(item for item in install_root.rglob("*") if item.is_file()):
        relative = path.relative_to(install_root).as_posix()
        if relative in ("runtime/bootstrapper.log",) or path.name == "engine_log.txt":
            continue
        size = path.stat().st_size
        total += size
        files.append({"path": relative, "size_bytes": size, "sha256": sha256_file(path)})
    return files, total


def verify(
    archive: Path,
    manifest_path: Path,
    install_root: Path,
    artifact_id: str,
) -> dict[str, Any]:
    if platform.system() not in ("Windows", "Darwin"):
        raise CpuBaseSmokeError(
            "CPU-base clean-machine verification currently requires Windows or macOS"
        )
    archive = archive.resolve()
    manifest_path = manifest_path.resolve()
    install_root = install_root.resolve()
    manifest = load_manifest(manifest_path, archive)
    base = install(archive, manifest, install_root)
    files, installed_size = package_inventory(install_root)

    suffix = ".exe" if os.name == "nt" else ""
    bootstrapper = install_root / f"cyxwiz-runtime-bootstrapper{suffix}"
    engine_check, engine_output = run_checked(
        bootstrapper,
        ["--runtime-root", str(install_root / "runtime"), "--package-smoke"],
        contaminated_environment(install_root),
        60,
    )
    parse_engine_smoke(engine_output)

    probe = base / f"cyxwiz-route-probe{suffix}"
    if not probe.is_file():
        raise CpuBaseSmokeError("CPU base is missing its route probe")
    checks = [engine_check]
    module_paths: list[str] = []
    benchmark: dict[str, Any] = {}
    for operation in OPERATIONS:
        check, output = run_checked(
            probe,
            ["--backend", "cpu", "--device", "0", "--operation", operation],
            package_environment(install_root, base),
            120,
        )
        parse_probe(output, operation)
        if operation == "constant" and platform.system() == "Windows":
            module_paths = audit_runtime_modules(output, base)
        if operation == "dense_compute_benchmark":
            timing = re.search(r"median_iteration_ms=([0-9.]+)", output)
            if timing is None:
                raise CpuBaseSmokeError("Dense CPU benchmark did not report a timing")
            benchmark = {
                "benchmark_id": "dense-compute-v1",
                "median_iteration_ms": float(timing.group(1)),
            }
        checks.append(check)

    if platform.system() == "Darwin":
        module_paths = audit_macos_dependencies(base)
    signed = manifest["signed"]
    return {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "artifact_kind": f"{signed['platform']}_cpu_base",
        "publication_status": (
            "signed" if manifest["signatures"] else "ci_unsigned_signing_request"
        ),
        "host": {"system": platform.system(), "machine": platform.machine()},
        "pack_id": signed["pack_id"],
        "runtime_set_id": signed["runtime_set_id"],
        "arrayfire_version": signed["arrayfire"]["version"],
        "archive": signed["archive"],
        "installed_size_bytes": installed_size,
        "file_count": len(files),
        "files": files,
        "development_path_isolation": {
            "status": "passed",
            "contaminated_variables": ["PATH", *OVERRIDE_VARIABLES],
            "package_runtime_modules": module_paths,
        },
        "checks": checks,
        "cpu_benchmark": benchmark,
        "accelerator_routes": [
            {"route": route, "status": "not_run", "reason": "physical_supported_hardware_required"}
            for route in ("cuda", "opencl", "oneapi")
        ],
        "result": "passed",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--install-root", required=True, type=Path)
    parser.add_argument("--artifact-id", default="native-cpu-base")
    parser.add_argument("--evidence", type=Path)
    arguments = parser.parse_args()
    try:
        evidence = verify(
            arguments.archive,
            arguments.manifest,
            arguments.install_root,
            arguments.artifact_id,
        )
        if arguments.evidence:
            arguments.evidence.parent.mkdir(parents=True, exist_ok=True)
            arguments.evidence.write_text(
                json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
            )
            print(f"CPU-base evidence written: {arguments.evidence}")
        print(f"Native CPU-base package smoke passed: {arguments.artifact_id}")
    except (
        ContractError,
        CpuBaseSmokeError,
        OSError,
        subprocess.SubprocessError,
        zipfile.BadZipFile,
    ) as error:
        print(f"CPU-base package smoke failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
