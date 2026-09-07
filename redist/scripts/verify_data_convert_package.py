#!/usr/bin/env python3
"""Check staged Windows DataConvert DLL closure and run local-fixture tests.

Uses only the staged DLLs and Windows System32 at runtime. This is a local
deployment smoke, not a network-disabled VM or a clean-machine qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

TESTS = (
    "test_excel_table_adapter", "test_data_convert_service",
    "test_data_convert_reload", "test_data_convert_formats",
    "test_data_convert_format_controls",
)
XLSX_NOTICES = ("openxlsx", "nowide", "pugixml", "libarchive", "zlib")


def imported_dlls(output: str) -> list[str]:
    return sorted(set(re.findall(r"^\s+([\w.+-]+\.dll)\s*$", output, re.I | re.M)))


def audit_dependencies(binaries: list[Path], stage: Path, dumpbin: Path) -> dict:
    system32 = Path(os.environ["SystemRoot"]) / "System32"
    pending = list(binaries)
    visited: set[Path] = set()
    bundled: dict[str, str] = {}
    prerequisites: set[str] = set()
    while pending:
        binary = pending.pop().resolve()
        if binary in visited:
            continue
        visited.add(binary)
        result = subprocess.run([str(dumpbin), "/DEPENDENTS", str(binary)],
                                capture_output=True, text=True, check=True, timeout=30)
        for name in imported_dlls(result.stdout):
            lower = name.lower()
            if re.fullmatch(r"(?:ucrtbased|msvcp\d+d|vcruntime[\d_]+d)\.dll", lower):
                raise RuntimeError(f"Debug CRT dependency in {binary.name}: {name}")
            if lower.startswith(("api-ms-", "ext-ms-")):
                continue  # Windows API-set contracts, not redistributable files.
            candidate = stage / name
            if candidate.is_file():
                if name not in bundled:
                    digest = hashlib.sha256()
                    with candidate.open("rb") as stream:
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(chunk)
                    bundled[name] = digest.hexdigest()
                pending.append(candidate)
            elif (system32 / name).is_file():
                prerequisites.add(name)
            else:
                raise RuntimeError(f"Missing staged dependency: {binary.name} -> {name}")
    return {"bundled_dll_sha256": bundled, "system_prerequisites": sorted(prerequisites)}


def verify(stage: Path, tests: Path, dumpbin: Path) -> dict:
    if os.name != "nt":
        raise RuntimeError("This verifier currently qualifies Windows packages only")
    for package in XLSX_NOTICES:
        matches = list((stage / "THIRD_PARTY_LICENSES/vcpkg").glob(f"*-{package}.txt"))
        if not matches or not any(path.stat().st_size for path in matches):
            raise RuntimeError(f"Missing/empty packaged XLSX notice: {package}")
    for name in ("OpenXLSX.dll", "nowide.dll", "pugixml.dll", "archive.dll", "zlib1.dll"):
        if not (stage / name).is_file():
            raise RuntimeError(f"Missing packaged XLSX DLL: {name}")
    binaries = [tests / f"{name}.exe" for name in TESTS]
    if any(not path.is_file() for path in binaries):
        raise RuntimeError("Build the Release DataConvert standalone test suite first")
    evidence = audit_dependencies(binaries, stage, dumpbin)
    environment = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "AF_PATH", "AF_PLUGIN_PATH",
                 "HDF5_PLUGIN_PATH", "CYXWIZ_ARRAYFIRE_DIR"):
        environment.pop(name, None)
    environment["PATH"] = str(stage) + os.pathsep + str(Path(os.environ["SystemRoot"]) / "System32")
    evidence.update({"stage": str(stage), "network_disabled": False,
                     "mode": "local-fixtures-with-sanitized-path", "tests": []})
    with tempfile.TemporaryDirectory(prefix="cyxwiz-dataconvert-package-") as temporary:
        work = Path(temporary)
        for binary in binaries:
            executable = work / binary.name
            shutil.copy2(binary, executable)
            result = subprocess.run([str(executable)], cwd=work, env=environment,
                                    capture_output=True, text=True, timeout=120)
            evidence["tests"].append({"name": binary.stem, "exit_code": result.returncode,
                                      "output": result.stdout + result.stderr})
            if result.returncode != 0:
                raise RuntimeError(f"{binary.name} failed: {result.returncode}\n{result.stdout}{result.stderr}")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--tests-dir", required=True, type=Path)
    parser.add_argument("--dumpbin", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    evidence = verify(args.stage.resolve(), args.tests_dir.resolve(), args.dumpbin.resolve())
    args.report.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    print(f"DataConvert staged-runtime smoke passed: {len(evidence['tests'])} tests; "
          f"{len(evidence['bundled_dll_sha256'])} bundled DLLs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
