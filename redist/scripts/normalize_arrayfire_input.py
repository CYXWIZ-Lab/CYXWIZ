"""Validate and stage POSIX ArrayFire redistribution inputs."""

import argparse
from pathlib import Path
import shutil


def discover(root: Path) -> tuple[Path, list[Path]]:
    root = root.resolve(strict=True)
    if not (root / "include" / "af" / "version.h").is_file():
        raise ValueError(f"Missing ArrayFire include/af/version.h under {root}")
    libraries = None
    for candidate in (root / "lib", root / "lib64"):
        if not candidate.is_dir():
            continue
        unified = any(p.is_file() for pattern in ("libaf.so*", "libaf.dylib", "libaf.*.dylib")
                      for p in candidate.glob(pattern))
        cpu = any(p.is_file() for pattern in ("libafcpu.so*", "libafcpu*.dylib")
                  for p in candidate.glob(pattern))
        if unified and cpu:
            libraries = candidate
            break
    if libraries is None:
        raise ValueError(f"Missing ArrayFire unified/CPU libraries in {root}/lib or lib64")
    notices = []
    directory = root / "LICENSES"
    if directory.is_dir() and any(p.is_file() for p in directory.rglob("*")):
        notices.append(directory)
    notices.extend(p for p in (root / "LICENSE", root / "LICENSE.txt") if p.is_file())
    if not notices:
        raise ValueError(f"Missing ArrayFire notices: expected nonempty {directory}, LICENSE, or LICENSE.txt")
    print(f"ArrayFire headers: {root / 'include'}")
    print(f"ArrayFire libraries: {libraries}")
    print(f"ArrayFire notices: {', '.join(str(p) for p in notices)}")
    return libraries, notices


def normalize(root: Path, destination: Path) -> None:
    libraries, notices = discover(root)
    destination = destination.resolve()
    root = root.resolve()
    if destination == root or root in destination.parents or destination in root.parents:
        raise ValueError("ArrayFire source and staging destination must not overlap")
    if destination.exists():
        raise ValueError(f"ArrayFire staging destination already exists: {destination}")
    shutil.copytree(root / "include", destination / "include")
    shutil.copytree(libraries, destination / "lib", symlinks=True)
    notice_destination = destination / "LICENSES"
    notice_destination.mkdir()
    for notice in notices:
        if notice.is_dir():
            shutil.copytree(notice, notice_destination, dirs_exist_ok=True)
        else:
            shutil.copy2(notice, notice_destination / notice.name)
    print(f"Staged ArrayFire redistribution inputs: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    try:
        if args.destination is None:
            discover(args.root)
        else:
            normalize(args.root, args.destination)
    except (OSError, ValueError) as error:
        parser.exit(1, f"ArrayFire redistribution input error: {error}\n")


if __name__ == "__main__":
    main()
