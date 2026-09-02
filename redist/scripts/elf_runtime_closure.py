#!/usr/bin/env python3
"""Close non-system ELF dependencies into a relocatable package stage."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
from typing import Callable, Sequence


class ElfClosureError(RuntimeError):
    pass


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class PackagedBinary:
    path: Path
    source: Path


@dataclass(frozen=True)
class ElfDependency:
    name: str
    path: Path | None


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )


def is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(4) == b"\x7fELF"
    except OSError:
        return False


def parse_ldd_dependencies(output: str) -> tuple[ElfDependency, ...]:
    dependencies: list[ElfDependency] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("linux-vdso"):
            continue
        missing = re.match(r"^(\S+)\s+=>\s+not found$", line)
        if missing:
            dependencies.append(ElfDependency(missing.group(1), None))
            continue
        mapped = re.match(r"^(\S+)\s+=>\s+(\S+)(?:\s+\(|$)", line)
        if mapped:
            dependencies.append(
                ElfDependency(mapped.group(1), Path(mapped.group(2)))
            )
            continue
        direct = re.match(r"^(/\S+)(?:\s+\(|$)", line)
        if direct:
            path = Path(direct.group(1))
            dependencies.append(ElfDependency(path.name, path))
    return tuple(dependencies)


def is_linux_system_library(path: Path) -> bool:
    value = path.as_posix()
    return any(
        value == root or value.startswith(root + "/")
        for root in ("/lib", "/lib64", "/usr/lib", "/usr/lib64")
    )


def packaged_rpath(binary: Path, stage: Path) -> str:
    directories = (stage, stage / "lib", stage / "arrayfire" / "lib")
    values = ["$ORIGIN"]
    for directory in directories:
        relative = os.path.relpath(directory, binary.parent).replace(os.sep, "/")
        value = "$ORIGIN" if relative == "." else f"$ORIGIN/{relative}"
        if value not in values:
            values.append(value)
    return ":".join(values)


def _dependencies(binary: Path, runner: CommandRunner) -> tuple[ElfDependency, ...]:
    result = runner(("ldd", str(binary)))
    if result.returncode != 0:
        raise ElfClosureError(
            f"Cannot inspect ELF dependencies for {binary}: {result.stdout.strip()}"
        )
    dependencies = parse_ldd_dependencies(result.stdout)
    missing = [item.name for item in dependencies if item.path is None]
    if missing:
        raise ElfClosureError(
            f"Unresolved ELF dependencies for {binary}: {', '.join(missing)}"
        )
    return dependencies


def _initial_binaries(stage: Path) -> list[PackagedBinary]:
    return [
        PackagedBinary(path.resolve(), path.resolve())
        for path in sorted(item for item in stage.rglob("*") if item.is_file())
        if is_elf(path)
    ]


def close_linux_runtime(
    stage: Path,
    *,
    runner: CommandRunner = _run,
) -> list[Path]:
    """Copy non-system ELF dependencies and assign package-relative RUNPATHs."""
    stage = stage.resolve()
    queue = _initial_binaries(stage)
    packaged_by_name: dict[str, Path] = {}
    for binary in queue:
        existing = packaged_by_name.setdefault(binary.path.name, binary.path)
        if existing != binary.path:
            raise ElfClosureError(
                f"Duplicate packaged ELF name {binary.path.name}: "
                f"{existing} and {binary.path}"
            )

    copied: list[Path] = []
    processed: set[Path] = set()
    while queue:
        binary = queue.pop(0)
        if binary.path in processed:
            continue
        processed.add(binary.path)
        for dependency in _dependencies(binary.source, runner):
            assert dependency.path is not None
            source = dependency.path
            if is_linux_system_library(source):
                continue
            target = packaged_by_name.get(dependency.name)
            if target is None:
                target = stage / "lib" / dependency.name
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(source, target, follow_symlinks=True)
                except OSError as error:
                    raise ElfClosureError(
                        f"Cannot copy ELF dependency {source} to {target}: {error}"
                    ) from error
                target.chmod(target.stat().st_mode | 0o755)
                target = target.resolve()
                packaged_by_name[dependency.name] = target
                copied.append(target)
                queue.append(PackagedBinary(target, source))

        result = runner(
            (
                "patchelf",
                "--set-rpath",
                packaged_rpath(binary.path, stage),
                str(binary.path),
            )
        )
        if result.returncode != 0:
            raise ElfClosureError(
                f"Cannot assign package-local RUNPATH for {binary.path}: "
                f"{result.stdout.strip()}"
            )
    return copied
