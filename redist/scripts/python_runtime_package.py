"""Copy only production Python runtime content into CyxWiz packages."""

from __future__ import annotations

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
