"""Copy only production Python runtime content into CyxWiz packages."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path, PurePath


_CACHE_DIRECTORY_NAMES = {
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}
_TEST_DIRECTORY_NAMES = {"test", "tests"}


def excludes_python_runtime_path(relative_path: PurePath) -> bool:
    """Return whether a Python-tree path is development-only content."""
    parts = tuple(part.casefold() for part in relative_path.parts)
    if not parts:
        return False
    if any(part in _CACHE_DIRECTORY_NAMES for part in parts):
        return True
    if relative_path.suffix.casefold() == ".pyc":
        return True

    in_library_tree = "site-packages" in parts or parts[0] in {"lib", "lib64"}
    return in_library_tree and any(
        part in _TEST_DIRECTORY_NAMES for part in parts
    )


def copy_python_runtime(source: Path, destination: Path) -> None:
    """Replace destination with a production-only copy of source."""
    source = source.resolve()
    if destination.exists():
        shutil.rmtree(destination)

    def ignored_names(directory: str, names: list[str]) -> set[str]:
        relative_directory = Path(directory).resolve().relative_to(source)
        return {
            name
            for name in names
            if excludes_python_runtime_path(relative_directory / name)
        }

    shutil.copytree(source, destination, ignore=ignored_names)


def excludes_standalone_python_path(relative_path: PurePath, system: str) -> bool:
    """Return whether a python-build-standalone path stays out of packages.

    Besides development content, POSIX trees drop share/ (man pages,
    terminfo), pkg-config files and every bin/ entry except the versioned
    interpreter: the helper scripts carry absolute build-machine shebangs.
    """
    if excludes_python_runtime_path(relative_path):
        return True
    if relative_path.suffix.casefold() == ".pdb":
        return True
    if system == "windows":
        return False
    parts = relative_path.parts
    lowered = tuple(part.casefold() for part in parts)
    if not lowered:
        return False
    if lowered[0] == "share" or lowered[:2] == ("lib", "pkgconfig"):
        return True
    return (
        lowered[0] == "bin"
        and len(parts) == 2
        and re.fullmatch(r"python3\.\d+", parts[1]) is None
    )


def copy_standalone_python(source: Path, destination: Path, system: str) -> None:
    """Replace destination with the packaged part of a standalone Python.

    POSIX symlinks are aliases (python3, libpython3.12.so); packages store
    regular files only, so they are left out rather than duplicated.
    """
    source = source.resolve()
    if destination.exists():
        shutil.rmtree(destination)

    def ignored_names(directory: str, names: list[str]) -> set[str]:
        relative_directory = Path(directory).resolve().relative_to(source)
        ignored = set()
        for name in names:
            if excludes_standalone_python_path(relative_directory / name, system):
                ignored.add(name)
            elif system != "windows" and os.path.islink(os.path.join(directory, name)):
                ignored.add(name)
        return ignored

    shutil.copytree(source, destination, ignore=ignored_names)
