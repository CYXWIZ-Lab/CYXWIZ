#!/usr/bin/env python3
"""Close and relocate non-system Mach-O dependencies into a package stage."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Callable, Iterable, Sequence


class MachOClosureError(RuntimeError):
    pass


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]

MACHO_MAGICS = {
    b"\xfe\xed\xfa\xce",  # 32-bit big endian
    b"\xce\xfa\xed\xfe",  # 32-bit little endian
    b"\xfe\xed\xfa\xcf",  # 64-bit big endian
    b"\xcf\xfa\xed\xfe",  # 64-bit little endian
    b"\xca\xfe\xba\xbe",  # universal binary
    b"\xbe\xba\xfe\xca",  # reverse universal binary
    b"\xca\xfe\xba\xbf",  # universal binary with 64-bit offsets
    b"\xbf\xba\xfe\xca",  # reverse universal binary with 64-bit offsets
}


@dataclass(frozen=True)
class PackagedBinary:
    path: Path
    source: Path


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=30,
    )


def parse_otool_dependencies(output: str) -> tuple[str, ...]:
    dependencies: list[str] = []
    for line in output.splitlines()[1:]:
        value = line.strip().split(" (compatibility version", 1)[0].strip()
        if value:
            dependencies.append(value)
    return tuple(dependencies)


def parse_otool_rpaths(output: str) -> tuple[str, ...]:
    lines = output.splitlines()
    values: list[str] = []
    for index, line in enumerate(lines):
        if line.strip() != "cmd LC_RPATH":
            continue
        for detail in lines[index + 1:index + 5]:
            stripped = detail.strip()
            if stripped.startswith("path "):
                values.append(stripped[5:].split(" (offset", 1)[0])
                break
    return tuple(values)


def is_system_dependency(value: str) -> bool:
    return value.startswith(("/System/Library/", "/usr/lib/"))


def is_macho(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(4) in MACHO_MAGICS
    except OSError:
        return False


def packaged_reference(owner: Path, dependency: Path) -> str:
    relative = os.path.relpath(dependency, owner.parent).replace(os.sep, "/")
    return f"@loader_path/{relative}"


def _inspect(
    binary: Path,
    option: str,
    runner: CommandRunner,
) -> str:
    result = runner(("otool", option, str(binary)))
    if result.returncode != 0:
        raise MachOClosureError(
            f"Cannot inspect Mach-O dependency data for {binary}: "
            f"{result.stdout.strip()}"
        )
    return result.stdout


def _expand_runtime_path(value: str, binary: PackagedBinary) -> Path | None:
    if value.startswith("@loader_path/"):
        return (binary.source.parent / value.removeprefix("@loader_path/")).resolve()
    if value.startswith("@executable_path/"):
        return None
    if value.startswith("/"):
        return Path(value).resolve()
    return None


def _resolve_dependency(
    value: str,
    binary: PackagedBinary,
    rpaths: Iterable[str],
    search_roots: Sequence[Path],
) -> Path:
    direct = _expand_runtime_path(value, binary)
    if direct is not None and direct.is_file():
        return direct

    if value.startswith("@rpath/"):
        suffix = value.removeprefix("@rpath/")
        for rpath in rpaths:
            expanded = _expand_runtime_path(rpath, binary)
            if expanded is not None:
                candidate = expanded / suffix
                if candidate.is_file():
                    return candidate.resolve()

    name = Path(value).name
    source_sibling = binary.source.parent / name
    if source_sibling.is_file():
        return source_sibling.resolve()

    matches: list[Path] = []
    for root in search_roots:
        candidate = root / name
        if candidate.is_file():
            matches.append(candidate.resolve())
    unique = list(dict.fromkeys(matches))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise MachOClosureError(
            f"Ambiguous Mach-O dependency {value!r} for {binary.path}: "
            + ", ".join(map(str, unique))
        )
    raise MachOClosureError(
        f"Unresolved non-system Mach-O dependency {value!r} for {binary.path}"
    )


def _initial_binaries(stage: Path) -> list[PackagedBinary]:
    binaries: list[PackagedBinary] = []
    for path in sorted(item for item in stage.rglob("*") if item.is_file()):
        if is_macho(path):
            binaries.append(PackagedBinary(path.resolve(), path.resolve()))
    return binaries


def close_macos_runtime(
    stage: Path,
    search_roots: Sequence[Path],
    *,
    runner: CommandRunner = _run,
) -> list[Path]:
    """Copy dependencies into ``stage/lib`` and rewrite load commands.

    System libraries and frameworks remain host prerequisites. Every other
    dependency must resolve through an explicit source/search root and is
    copied into the signed package boundary.
    """
    stage = stage.resolve()
    roots = tuple(path.resolve() for path in search_roots if path.is_dir())
    queue = _initial_binaries(stage)
    packaged_by_source = {item.source: item.path for item in queue}
    packaged_by_name: dict[str, Path] = {}
    for item in queue:
        existing = packaged_by_name.setdefault(item.path.name, item.path)
        if existing != item.path:
            raise MachOClosureError(
                f"Duplicate packaged Mach-O name {item.path.name}: "
                f"{existing} and {item.path}"
            )

    copied: list[Path] = []
    processed: set[Path] = set()
    while queue:
        binary = queue.pop(0)
        if binary.path in processed:
            continue
        processed.add(binary.path)
        dependencies = parse_otool_dependencies(_inspect(binary.path, "-L", runner))
        rpaths = parse_otool_rpaths(_inspect(binary.path, "-l", runner))
        for dependency in dependencies:
            if is_system_dependency(dependency):
                continue
            if (
                binary.path.suffix == ".dylib"
                and Path(dependency).name == binary.path.name
            ):
                continue
            source = _resolve_dependency(dependency, binary, rpaths, roots)
            target = packaged_by_source.get(source)
            if target is None:
                target = packaged_by_name.get(source.name)
                if target is not None:
                    packaged_by_source[source] = target
            if target is None:
                target = stage / "lib" / source.name
                collision = packaged_by_name.get(target.name)
                if collision is not None and collision != target:
                    raise MachOClosureError(
                        f"Mach-O dependency name collision for {target.name}: "
                        f"{collision} and {source}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target, follow_symlinks=True)
                target.chmod(target.stat().st_mode | 0o755)
                target = target.resolve()
                packaged_by_source[source] = target
                packaged_by_name[target.name] = target
                copied.append(target)
                queue.append(PackagedBinary(target, source))
            replacement = packaged_reference(binary.path, target)
            result = runner(
                ("install_name_tool", "-change", dependency, replacement, str(binary.path))
            )
            if result.returncode != 0:
                raise MachOClosureError(
                    f"Cannot relocate {dependency!r} in {binary.path}: "
                    f"{result.stdout.strip()}"
                )

        if binary.path.suffix == ".dylib":
            result = runner(
                ("install_name_tool", "-id", f"@rpath/{binary.path.name}", str(binary.path))
            )
            if result.returncode != 0:
                raise MachOClosureError(
                    f"Cannot assign package-local install name for {binary.path}: "
                    f"{result.stdout.strip()}"
                )
    return copied
