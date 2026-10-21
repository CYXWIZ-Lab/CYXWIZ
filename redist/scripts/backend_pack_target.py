#!/usr/bin/env python3
"""Canonical native targets supported by CyxWiz backend-pack releases."""

from __future__ import annotations

from dataclasses import dataclass
import platform


class BackendPackTargetError(ValueError):
    """Raised when a host cannot produce a supported native release target."""


@dataclass(frozen=True)
class BackendPackTarget:
    system: str
    platform: str
    architecture: str
    executable_suffix: str
    library_suffix: str

    @property
    def artifact_suffix(self) -> str:
        return f"{self.platform}-{self.architecture}"


_SYSTEMS = {
    "windows": ("win64", ".exe", ".dll", {"x86_64"}),
    "linux": ("linux64", "", ".so", {"x86_64"}),
    "darwin": ("macos", "", ".dylib", {"x86_64", "arm64"}),
}

_ARCHITECTURES = {
    "amd64": "x86_64",
    "x64": "x86_64",
    "x86_64": "x86_64",
    "aarch64": "arm64",
    "arm64": "arm64",
}


def resolve_backend_pack_target(system: str, machine: str) -> BackendPackTarget:
    system_id = system.strip().lower()
    machine_id = machine.strip().lower()
    configuration = _SYSTEMS.get(system_id)
    architecture = _ARCHITECTURES.get(machine_id)
    if configuration is None or architecture is None:
        raise BackendPackTargetError(
            f"Unsupported packaging host: {system or '<empty>'}/{machine or '<empty>'}"
        )
    platform_id, executable_suffix, library_suffix, architectures = configuration
    if architecture not in architectures:
        raise BackendPackTargetError(
            f"Unsupported release target: {platform_id}/{architecture}"
        )
    return BackendPackTarget(
        system=system_id,
        platform=platform_id,
        architecture=architecture,
        executable_suffix=executable_suffix,
        library_suffix=library_suffix,
    )


def detect_backend_pack_target() -> BackendPackTarget:
    return resolve_backend_pack_target(platform.system(), platform.machine())
