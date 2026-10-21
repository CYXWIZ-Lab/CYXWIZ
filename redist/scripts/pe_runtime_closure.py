#!/usr/bin/env python3
"""Read Windows PE imports and audit the app-local MSVC runtime closure."""

from __future__ import annotations

from pathlib import Path
import re
import struct
from typing import Callable, Iterable

PE_SUFFIXES = (".exe", ".dll", ".pyd")
MSVC_RUNTIME = re.compile(
    r"(?:vcruntime140(?:_1)?|msvcp140(?:_[a-z0-9_]+)?|concrt140|vcomp140)\.dll",
    re.IGNORECASE,
)


# GPU runtimes belong to optional packs or vendor drivers; the portable base
# must reach them only through ArrayFire's runtime plugin loading.
GPU_RUNTIME = re.compile(
    r"(?:af(?:cuda|opencl|oneapi)|opencl|nvcuda|cudart64_\d+|cublas(?:lt)?64_\d+"
    r"|nvrtc64_[0-9_]+|sycl\d*)\.dll",
    re.IGNORECASE,
)


class PeFormatError(ValueError):
    pass


def is_pe(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            header = stream.read(0x40)
            if len(header) < 0x40 or header[:2] != b"MZ":
                return False
            stream.seek(struct.unpack_from("<I", header, 0x3C)[0])
            return stream.read(4) == b"PE\0\0"
    except OSError:
        return False


def pe_imported_dlls(path: Path) -> list[str]:
    """Return the DLL names in the PE import directory (not delay-load)."""
    data = path.read_bytes()

    def u16(offset: int) -> int:
        return struct.unpack_from("<H", data, offset)[0]

    def u32(offset: int) -> int:
        return struct.unpack_from("<I", data, offset)[0]

    try:
        pe = u32(0x3C)
        if data[pe:pe + 4] != b"PE\0\0":
            raise PeFormatError(f"{path.name} is not a PE image")
        section_count = u16(pe + 6)
        optional = pe + 24
        magic = u16(optional)
        if magic == 0x20B:
            directories = optional + 112
        elif magic == 0x10B:
            directories = optional + 96
        else:
            raise PeFormatError(f"{path.name} has an unknown optional header")
        import_rva = u32(directories + 8)
        sections = optional + u16(pe + 20)
        table = [
            (u32(base + 12), max(u32(base + 8), u32(base + 16)), u32(base + 20))
            for base in (sections + 40 * index for index in range(section_count))
        ]
    except struct.error as error:
        raise PeFormatError(f"{path.name} has a truncated PE header") from error

    def offset_of(rva: int) -> int:
        for virtual, size, raw in table:
            if virtual <= rva < virtual + size:
                return rva - virtual + raw
        raise PeFormatError(f"{path.name} import RVA {rva:#x} is outside sections")

    if import_rva == 0:
        return []
    names: list[str] = []
    descriptor = offset_of(import_rva)
    try:
        while True:
            name_rva = u32(descriptor + 12)
            if name_rva == 0 and u32(descriptor) == 0:
                break
            start = offset_of(name_rva)
            end = data.index(b"\0", start)
            names.append(data[start:end].decode("ascii"))
            descriptor += 20
    except (struct.error, ValueError, UnicodeDecodeError) as error:
        raise PeFormatError(f"{path.name} has a malformed import table") from error
    return names


def audit_gpu_runtime_imports(
    root: Path,
    read_imports: Callable[[Path], Iterable[str]] = pe_imported_dlls,
) -> list[str]:
    """Return images in a CPU base that link a GPU runtime directly."""
    problems: list[str] = []
    for image in sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in PE_SUFFIXES and is_pe(path)
    ):
        linked = sorted(
            {name.lower() for name in read_imports(image) if GPU_RUNTIME.fullmatch(name)}
        )
        if linked:
            relative = image.relative_to(root).as_posix()
            problems.append(f"{relative} imports {', '.join(linked)}")
    return problems


def audit_msvc_runtime_closure(
    root: Path,
    read_imports: Callable[[Path], Iterable[str]] = pe_imported_dlls,
) -> list[str]:
    """Return problems where an MSVC runtime import is not satisfied app-locally.

    An executable must find the runtime in its own directory (the first loader
    search location). A DLL or extension is loaded into such a process, so the
    runtime only has to exist somewhere in the package.
    """
    images = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in PE_SUFFIXES and is_pe(path)
    )
    shipped = {path.name.lower() for path in root.rglob("*.dll") if path.is_file()}
    problems: list[str] = []
    for image in images:
        needed = sorted(
            {name.lower() for name in read_imports(image) if MSVC_RUNTIME.fullmatch(name)}
        )
        if image.suffix.lower() == ".exe":
            beside = {path.name.lower() for path in image.parent.iterdir() if path.is_file()}
            missing = [name for name in needed if name not in beside]
        else:
            missing = [name for name in needed if name not in shipped]
        if missing:
            relative = image.relative_to(root).as_posix()
            problems.append(f"{relative} needs {', '.join(missing)}")
    return problems
