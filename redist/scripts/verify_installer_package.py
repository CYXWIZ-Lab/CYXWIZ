#!/usr/bin/env python3
"""Validate a staged standalone-installer package on a fresh machine."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys


class PackageSmokeError(RuntimeError):
    pass


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise PackageSmokeError(f"Required package file is missing: {path.name}")
    return path


def run_checked(path: Path, arguments: list[str], expected: int) -> None:
    environment = os.environ.copy()
    if os.name == "nt":
        windows = Path(environment.get("SystemRoot", r"C:\Windows"))
        environment["PATH"] = os.pathsep.join(
            (str(path.parent), str(windows / "System32"), str(windows))
        )
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
    if result.returncode != expected:
        output = result.stdout.strip()
        raise PackageSmokeError(
            f"{path.name} returned {result.returncode}, expected {expected}: {output}"
        )


def dependency_output(command: list[str]) -> str:
    result = subprocess.run(
        command,
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


def audit_dependencies(executables: list[Path]) -> None:
    system = platform.system()
    forbidden = ("vcpkg_installed", "/home/runner/work/", "/Users/runner/work/")
    for executable in executables:
        if system == "Linux":
            output = dependency_output(["ldd", str(executable)])
            if "not found" in output:
                raise PackageSmokeError(
                    f"{executable.name} has an unresolved dependency:\n{output}"
                )
        elif system == "Darwin":
            output = dependency_output(["otool", "-L", str(executable)])
            output = "\n".join(output.splitlines()[1:])
        else:
            continue
        if any(marker in output for marker in forbidden):
            raise PackageSmokeError(
                f"{executable.name} retains a build-machine dependency:\n{output}"
            )


def verify(stage: Path) -> None:
    stage = stage.resolve()
    suffix = ".exe" if os.name == "nt" else ""
    installer = require_file(stage / f"cyxwiz-installer{suffix}")
    helper = require_file(stage / f"cyxwiz-backend-pack-installer{suffix}")
    model_test = require_file(
        stage / "smoke" / f"test_backend_pack_manager_model{suffix}"
    )
    for executable in (installer, helper, model_test):
        executable.chmod(executable.stat().st_mode | 0o111)

    audit_dependencies([installer, helper])
    run_checked(installer, ["--package-smoke"], 0)
    run_checked(helper, [], 78)
    run_checked(model_test, [], 0)
    print(f"Installer package smoke passed: {stage}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=Path)
    arguments = parser.parse_args()
    try:
        verify(arguments.stage)
    except (OSError, subprocess.SubprocessError, PackageSmokeError) as error:
        print(f"Installer package smoke failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
