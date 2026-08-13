#!/usr/bin/env python3
"""Build truthful CyxWiz minimal and full redistribution archives."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


SUPPORTED_BACKENDS = ("cpu", "cuda", "oneapi", "opencl")


class PackageError(RuntimeError):
    pass


@dataclass(frozen=True)
class PackagePaths:
    root: Path
    redist: Path
    build: Path
    resources: Path
    templates: Path
    output_root: Path


@dataclass(frozen=True)
class Component:
    path: str
    size: int
    sha256: str
    source: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a validated CyxWiz redistribution package."
    )
    parser.add_argument("profile", choices=("minimal", "full"))
    parser.add_argument("--version", help="CyxWiz version; inferred when omitted")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--resources-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--arrayfire-dir", type=Path)
    parser.add_argument("--python-dir", type=Path)
    parser.add_argument("--python-version")
    parser.add_argument(
        "--intel-runtime-license-dir",
        type=Path,
        help="Notices for bundled Intel MKL/SYCL runtimes",
    )
    parser.add_argument(
        "--nvidia-runtime-license-dir",
        type=Path,
        help="Notices for bundled NVIDIA CUDA runtimes",
    )
    parser.add_argument(
        "--backends",
        default="cpu",
        help="Comma-separated full-package backends (cpu is always included)",
    )
    parser.add_argument(
        "--stage-only", action="store_true", help="Validate and stage without archiving"
    )
    return parser.parse_args(argv)


def host_platform() -> tuple[str, str, str]:
    system = platform.system().lower()
    if system == "windows":
        return "win64", ".exe", ".dll"
    if system == "linux":
        return "linux64", "", ".so"
    if system == "darwin":
        return "macos", "", ".dylib"
    raise PackageError(f"Unsupported packaging host: {platform.system()}")


def default_paths(script: Path, args: argparse.Namespace) -> PackagePaths:
    redist = script.resolve().parent.parent
    root = redist.parent
    return PackagePaths(
        root=root,
        redist=redist,
        build=(args.build_dir or root / "build" / "bin" / "Release").resolve(),
        resources=(args.resources_dir or root / "cyxwiz-engine" / "resources").resolve(),
        templates=(redist / "templates").resolve(),
        output_root=(args.output_root or redist / "output").resolve(),
    )


def parse_backends(value: str) -> tuple[str, ...]:
    requested = {item.strip().lower() for item in value.split(",") if item.strip()}
    unknown = requested.difference(SUPPORTED_BACKENDS)
    if unknown:
        raise PackageError(f"Unsupported backend(s): {', '.join(sorted(unknown))}")
    requested.add("cpu")
    return tuple(name for name in SUPPORTED_BACKENDS if name in requested)


def infer_cyxwiz_version(root: Path) -> str:
    header = root / "cyxwiz-backend" / "include" / "cyxwiz" / "cyxwiz.h"
    text = header.read_text(encoding="utf-8")
    values = {}
    for part in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(rf"#define\s+CYXWIZ_VERSION_{part}\s+(\d+)", text)
        if not match:
            raise PackageError("Pass --version; CyxWiz version could not be inferred")
        values[part] = match.group(1)
    return f"{values['MAJOR']}.{values['MINOR']}.{values['PATCH']}"


def arrayfire_version(arrayfire_root: Path) -> str:
    header = arrayfire_root / "include" / "af" / "version.h"
    if not header.is_file():
        return "unknown"
    match = re.search(r'#define\s+AF_VERSION\s+"([^"]+)"', header.read_text(encoding="utf-8"))
    return match.group(1) if match else "unknown"


def python_version(python_root: Path, explicit: str | None) -> str:
    candidates = [python_root / "python.exe"]
    candidates.extend((python_root / "bin").glob("python3*"))
    detected = None
    for executable in candidates:
        if not executable.is_file():
            continue
        try:
            output = subprocess.check_output(
                [str(executable), "--version"], text=True, stderr=subprocess.STDOUT
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            continue
        match = re.search(r"Python\s+([0-9.]+)", output)
        if match:
            detected = match.group(1)
            break
    if explicit:
        if detected and detected != explicit:
            raise PackageError(
                f"Declared Python version {explicit} does not match runtime {detected}"
            )
        return explicit
    if detected:
        return detected
    raise PackageError("Pass --python-version; bundled Python version could not be detected")


def validate_release_version(version: str, label: str) -> str:
    if not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z._-]*", version):
        raise PackageError(f"Invalid {label} version: {version!r}")
    return version


def validate_python_version(version: str) -> str:
    validate_release_version(version, "Python")
    if not re.fullmatch(r"3\.12(?:\.\d+)?", version):
        raise PackageError(f"Full packages require Python 3.12.x, found {version}")
    return version


def validate_windows_embedded_python(python_root: Path) -> None:
    for name, description in (
        ("python.exe", "bundled Python executable"),
        ("python312.dll", "Python 3.12 runtime DLL"),
        ("python312.zip", "Python 3.12 standard-library archive"),
        ("python312._pth", "Python embedded path configuration"),
    ):
        require_file(python_root / name, description)


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise PackageError(f"Missing {description}: {path}")
    return path


def require_directory(path: Path, description: str) -> Path:
    if not path.is_dir():
        raise PackageError(f"Missing {description}: {path}")
    return path


def require_notice_directory(path: Path, description: str) -> Path:
    require_directory(path, description)
    if not any(item.is_file() for item in path.rglob("*")):
        raise PackageError(f"{description} contains no notice files: {path}")
    return path


def validate_intel_runtime_notices(path: Path, backends: Sequence[str]) -> Path:
    require_notice_directory(path, "Intel runtime license directory")
    files = [item for item in path.rglob("*") if item.is_file()]

    def relative_text(item: Path) -> str:
        return item.relative_to(path).as_posix().lower()

    def has_notice(component: str, names: Sequence[str]) -> bool:
        return any(component in relative_text(item) and any(name in item.name.lower() for name in names) for item in files)

    missing = []
    if not has_notice("mkl", ("license", "eula")):
        missing.append("oneMKL license")
    if not has_notice("mkl", ("third-party", "third_party")):
        missing.append("oneMKL third-party programs")

    if "oneapi" in backends:
        compiler_files = [
            item
            for item in files
            if any(token in relative_text(item) for token in ("dpcpp", "compiler", "sycl"))
        ]
        if not any(item.name.lower() == "credist.txt" for item in compiler_files):
            missing.append("DPC++ redistributable list (credist.txt)")
        if not any(
            "third-party" in item.name.lower() or "third_party" in item.name.lower()
            for item in compiler_files
        ):
            missing.append("DPC++ third-party programs")
        if not any(
            "license" in item.name.lower() or "eula" in item.name.lower()
            for item in compiler_files
        ):
            missing.append("DPC++ license/EULA")

    if missing:
        raise PackageError(
            f"Incomplete Intel runtime notices in {path}; missing: {', '.join(missing)}"
        )
    return path


def safe_clean(stage: Path, output_root: Path) -> None:
    resolved_stage = stage.resolve()
    resolved_root = output_root.resolve()
    if resolved_stage == resolved_root or resolved_root not in resolved_stage.parents:
        raise PackageError(f"Refusing to clean unsafe staging path: {resolved_stage}")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def dynamic_library_matches(build: Path, suffix: str) -> list[Path]:
    if suffix == ".dll":
        return sorted(build.glob("*.dll"))
    return sorted(path for path in build.glob(f"*{suffix}*") if path.is_file())


def backend_runtime(paths: PackagePaths, lib_suffix: str) -> Path:
    names = [f"cyxwiz-backend{lib_suffix}", f"libcyxwiz-backend{lib_suffix}"]
    for name in names:
        candidate = paths.build / name
        if candidate.is_file():
            return candidate
    raise PackageError(f"Missing CyxWiz backend runtime in {paths.build}")


def copy_vcpkg_notices(paths: PackagePaths, stage: Path, system: str) -> None:
    roots = sorted((paths.root / "build" / "vcpkg_installed").glob("*/share"))
    notices = []
    for root in roots:
        notices.extend(sorted(root.glob("*/copyright")))
    if not notices:
        if system == "windows":
            raise PackageError("Missing vcpkg copyright notices below build/vcpkg_installed")
        return
    destination = stage / "THIRD_PARTY_LICENSES" / "vcpkg"
    destination.mkdir(parents=True, exist_ok=True)
    for notice in notices:
        triplet = notice.parents[1].parent.name
        package = notice.parent.name
        copy_file(notice, destination / f"{triplet}-{package}.txt")


def copy_runtime_notices(
    stage: Path,
    intel_notices: Path,
    nvidia_notices: Path | None,
) -> None:
    copy_tree(intel_notices, stage / "THIRD_PARTY_LICENSES" / "Intel")
    if nvidia_notices is not None:
        copy_tree(nvidia_notices, stage / "THIRD_PARTY_LICENSES" / "NVIDIA")


def copy_build_payload(
    paths: PackagePaths,
    stage: Path,
    profile: str,
    system: str,
    exe_suffix: str,
    lib_suffix: str,
) -> None:
    engine = require_file(paths.build / f"cyxwiz-engine{exe_suffix}", "Release Engine executable")
    route_probe = require_file(
        paths.build / f"cyxwiz-route-probe{exe_suffix}",
        "isolated compute-route qualification probe",
    )
    backend_runtime(paths, lib_suffix)
    require_directory(paths.resources, "Engine resources")

    copy_file(engine, stage / engine.name)
    copy_file(route_probe, stage / route_probe.name)
    for library in dynamic_library_matches(paths.build, lib_suffix):
        if re.fullmatch(r"python\d+\.dll", library.name, re.IGNORECASE):
            continue
        copy_file(library, stage / library.name)

    for pattern in ("*.pyd", "pycyxwiz*.so", "cyxwiz_plotting*.so", "*.dylib"):
        for binding in sorted(paths.build.glob(pattern)):
            if binding.is_file():
                copy_file(binding, stage / binding.name)

    plugins = paths.build / "plugins"
    if plugins.is_dir():
        copy_tree(plugins, stage / "plugins")
    copy_tree(paths.resources, stage / "resources")
    copy_file(require_file(paths.root / "LICENSE", "CyxWiz LICENSE"), stage / "LICENSE")
    copy_vcpkg_notices(paths, stage, system)


def arrayfire_library_dir(root: Path) -> Path:
    for candidate in (root / "lib", root / "bin", root):
        if candidate.is_dir():
            return candidate
    raise PackageError(f"ArrayFire library directory not found below {root}")


def first_matches(roots: Iterable[Path], pattern: str) -> list[Path]:
    matches: dict[str, Path] = {}
    for root in roots:
        if root.is_dir():
            for path in sorted(root.glob(pattern)):
                if path.is_file():
                    matches.setdefault(path.name.lower(), path)
    return list(matches.values())


def copy_required_group(roots: Sequence[Path], pattern: str, destination: Path, label: str) -> None:
    matches = first_matches(roots, pattern)
    if not matches:
        raise PackageError(f"Missing {label}; expected pattern {pattern} in {', '.join(map(str, roots))}")
    for source in matches:
        target = destination / source.name
        if not target.exists():
            copy_file(source, target)


def copy_optional_groups(roots: Sequence[Path], patterns: Sequence[str], destination: Path) -> None:
    copied: set[str] = set()
    for pattern in patterns:
        for source in first_matches(roots, pattern):
            key = source.name.lower()
            target = destination / source.name
            if key not in copied and not target.exists():
                copy_file(source, target)
                copied.add(key)


def package_arrayfire(
    arrayfire_root: Path,
    stage: Path,
    backends: Sequence[str],
    lib_suffix: str,
) -> str:
    require_directory(arrayfire_root, "ArrayFire root")
    library_dir = arrayfire_library_dir(arrayfire_root)
    destination = stage / "arrayfire" / ("bin" if lib_suffix == ".dll" else "lib")
    destination.mkdir(parents=True, exist_ok=True)
    prefix = "" if lib_suffix == ".dll" else "lib"
    suffix_pattern = f"{lib_suffix}*" if lib_suffix != ".dll" else lib_suffix
    roots = [library_dir]

    copy_required_group(roots, f"{prefix}af{suffix_pattern}", destination, "ArrayFire unified runtime")
    copy_required_group(roots, f"{prefix}afcpu{suffix_pattern}", destination, "ArrayFire CPU backend")

    if lib_suffix == ".dll":
        copy_required_group(roots, "mkl_rt*.dll", destination, "ArrayFire CPU MKL runtime")
        copy_optional_groups(
            roots,
            ("mkl_core*.dll", "mkl_def*.dll", "mkl_avx*.dll", "mkl_mc*.dll", "mkl_tbb_thread*.dll", "tbb*.dll", "libmmd.dll"),
            destination,
        )

    if "opencl" in backends:
        copy_required_group(roots, f"{prefix}afopencl{suffix_pattern}", destination, "ArrayFire OpenCL backend")

    if "cuda" in backends:
        copy_required_group(roots, f"{prefix}afcuda{suffix_pattern}", destination, "ArrayFire CUDA backend")
        if lib_suffix == ".dll":
            for pattern, label in (
                ("cublas64_*.dll", "CUDA cuBLAS runtime"),
                ("cufft64_*.dll", "CUDA cuFFT runtime"),
                ("cusolver64_*.dll", "CUDA cuSOLVER runtime"),
                ("nvrtc64_*.dll", "CUDA NVRTC runtime"),
            ):
                copy_required_group(roots, pattern, destination, label)
            copy_optional_groups(
                roots,
                ("cublasLt64_*.dll", "cusparse64_*.dll", "nvrtc-builtins*.dll", "nvJitLink*.dll", "cudnn*.dll"),
                destination,
            )

    if "oneapi" in backends:
        copy_required_group(roots, f"{prefix}afoneapi{suffix_pattern}", destination, "ArrayFire oneAPI backend")
        if lib_suffix == ".dll":
            for pattern, label in (
                ("sycl*.dll", "oneAPI SYCL runtime"),
                ("mkl_sycl_blas*.dll", "oneMKL SYCL BLAS runtime"),
                ("mkl_sycl_lapack*.dll", "oneMKL SYCL LAPACK runtime"),
                ("mkl_sycl_dft*.dll", "oneMKL SYCL DFT runtime"),
                ("mkl_sycl_sparse*.dll", "oneMKL SYCL sparse runtime"),
                ("ur_loader*.dll", "oneAPI Unified Runtime loader"),
                ("ur_adapter_*.dll", "oneAPI Unified Runtime adapter"),
                ("libmmd.dll", "Intel math runtime"),
            ):
                copy_required_group(roots, pattern, destination, label)
            copy_optional_groups(
                roots,
                ("mkl_sycl_*.dll", "mkl_*.dll", "ur_*.dll", "umf*.dll", "tbb*.dll", "libhwloc*.dll"),
                destination,
            )

    licenses = arrayfire_root / "LICENSES"
    require_directory(licenses, "ArrayFire licenses")
    copy_tree(licenses, stage / "THIRD_PARTY_LICENSES" / "ArrayFire")
    return arrayfire_version(arrayfire_root)


def create_launcher(stage: Path, profile: str, system: str) -> None:
    if system == "windows":
        arrayfire_path = (
            r'%CYXWIZ_HOME%arrayfire\bin'
            if profile == "full"
            else r'%CYXWIZ_ARRAYFIRE_DIR%'
        )
        lines = [
            "@echo off",
            "setlocal",
            'set "CYXWIZ_HOME=%~dp0"',
        ]
        if profile == "minimal":
            lines.extend(
                [
                    "if not defined CYXWIZ_ARRAYFIRE_DIR goto launch",
                    f'set "PATH={arrayfire_path};%PATH%"',
                    ":launch",
                ]
            )
        else:
            lines.extend(
                [
                    f'set "PATH=%CYXWIZ_HOME%;{arrayfire_path};%CYXWIZ_HOME%python;%PATH%"',
                ]
            )
        lines.extend(['"%CYXWIZ_HOME%cyxwiz-engine.exe" %*', "exit /b %errorlevel%", ""])
        (stage / "start_cyxwiz.bat").write_text("\r\n".join(lines), encoding="ascii")
        return

    loader = "DYLD_LIBRARY_PATH" if system == "darwin" else "LD_LIBRARY_PATH"
    extra = ':$SCRIPT_DIR/arrayfire/lib:$SCRIPT_DIR/python/lib' if profile == "full" else ""
    launcher = f'''#!/bin/sh
set -eu
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
export {loader}="$SCRIPT_DIR{extra}:${{{loader}:-}}"
exec "$SCRIPT_DIR/cyxwiz-engine" "$@"
'''
    path = stage / "cyxwiz"
    path.write_text(launcher, encoding="ascii")
    path.chmod(0o755)


def render_readme(template: Path, destination: Path, values: dict[str, str]) -> None:
    text = require_file(template, "README template").read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    unresolved = sorted(set(re.findall(r"{{[A-Z0-9_]+}}", text)))
    if unresolved:
        raise PackageError(f"Unresolved README template values: {', '.join(unresolved)}")
    destination.write_text(text, encoding="utf-8", newline="\n")


def classify_source(relative_path: Path) -> str:
    value = relative_path.as_posix()
    if value.startswith("arrayfire/") or value.startswith("THIRD_PARTY_LICENSES/ArrayFire/"):
        return "arrayfire"
    if value.startswith("THIRD_PARTY_LICENSES/Intel/"):
        return "intel-runtime-license"
    if value.startswith("THIRD_PARTY_LICENSES/NVIDIA/"):
        return "nvidia-runtime-license"
    if value.startswith("THIRD_PARTY_LICENSES/vcpkg/"):
        return "vcpkg-license"
    if value.startswith("python/"):
        return "python"
    if value.startswith("resources/"):
        return "cyxwiz-resources"
    if value in ("README.md", "LICENSE", "start_cyxwiz.bat", "cyxwiz"):
        return "cyxwiz-package"
    return "cyxwiz-build"


def inventory(stage: Path) -> list[Component]:
    components = []
    for path in sorted(item for item in stage.rglob("*") if item.is_file()):
        relative = path.relative_to(stage)
        digest_builder = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest_builder.update(chunk)
        digest = digest_builder.hexdigest()
        components.append(
            Component(relative.as_posix(), path.stat().st_size, digest, classify_source(relative))
        )
    return components


def write_manifest(
    stage: Path,
    profile: str,
    version: str,
    platform_name: str,
    backends: Sequence[str],
    versions: dict[str, str],
) -> Path:
    components = inventory(stage)
    prerequisites = [
        "64-bit supported operating system",
        "Microsoft Visual C++ 2015-2022 x64 Redistributable on Windows",
        "compatible hardware drivers/providers",
    ]
    if profile == "minimal":
        prerequisites.extend(["Python 3.12", "matching ArrayFire unified runtime and selected backend"])
    document = {
        "schema_version": 1,
        "package": {
            "product": "CyxWiz Engine",
            "version": version,
            "profile": profile,
            "platform": platform_name,
            "arrayfire_backends": list(backends),
            "generated_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        },
        "dependency_versions": versions,
        "external_prerequisites": prerequisites,
        "components": [component.__dict__ for component in components],
        "component_count": len(components),
        "payload_bytes": sum(component.size for component in components),
    }
    path = stage / "PACKAGE_MANIFEST.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def create_archive(stage: Path, output_root: Path, package_name: str, system: str) -> Path:
    base = output_root / package_name
    if system == "windows":
        archive = Path(shutil.make_archive(str(base), "zip", root_dir=stage))
    else:
        archive = Path(shutil.make_archive(str(base), "gztar", root_dir=stage))
    return archive


def build_package(args: argparse.Namespace, script: Path | None = None) -> tuple[Path, Path | None]:
    script = script or Path(__file__)
    paths = default_paths(script, args)
    platform_name, exe_suffix, lib_suffix = host_platform()
    system = platform.system().lower()
    version = validate_release_version(args.version or infer_cyxwiz_version(paths.root), "CyxWiz")
    backends = parse_backends(args.backends) if args.profile == "full" else ()
    if args.profile == "full" and system != "windows":
        raise PackageError(
            "Self-contained full packaging is currently validated only on Windows; "
            "use the minimal profile until the platform runtime closure is defined"
        )
    arrayfire_root = None
    python_root = None
    full_python_version = None
    intel_notices = None
    nvidia_notices = None
    if args.profile == "full":
        arrayfire_root = args.arrayfire_dir or Path(
            os.environ.get(
                "ARRAYFIRE_DIR",
                r"C:\Program Files\ArrayFire\v3" if system == "windows" else "/usr/local",
            )
        )
        arrayfire_root = arrayfire_root.resolve()
        require_directory(arrayfire_root, "ArrayFire root")

        default_python = Path(r"C:\Python312-embed" if system == "windows" else "/opt/python3.12")
        python_root = (args.python_dir or Path(os.environ.get("PYTHON_EMBED", default_python))).resolve()
        require_directory(python_root, "bundled Python runtime")
        validate_windows_embedded_python(python_root)
        python_license = next(
            (path for path in (python_root / "LICENSE.txt", python_root / "LICENSE") if path.is_file()),
            None,
        )
        if python_license is None:
            raise PackageError(f"Missing bundled Python license in {python_root}")
        full_python_version = validate_python_version(
            python_version(python_root, args.python_version)
        )
        intel_notices = args.intel_runtime_license_dir or os.environ.get(
            "INTEL_RUNTIME_LICENSE_DIR"
        )
        if intel_notices is None:
            raise PackageError(
                "Pass --intel-runtime-license-dir for bundled MKL/SYCL redistribution notices"
            )
        intel_notices = validate_intel_runtime_notices(
            Path(intel_notices).resolve(), backends
        )
        if "cuda" in backends:
            nvidia_notices = args.nvidia_runtime_license_dir or os.environ.get(
                "NVIDIA_RUNTIME_LICENSE_DIR"
            )
            if nvidia_notices is None:
                raise PackageError(
                    "Pass --nvidia-runtime-license-dir for bundled CUDA redistribution notices"
                )
            nvidia_notices = require_notice_directory(
                Path(nvidia_notices).resolve(), "NVIDIA runtime license directory"
            )
    stage_name = "minimal" if args.profile == "minimal" else "full"
    stage = paths.output_root / stage_name
    safe_clean(stage, paths.output_root)

    copy_build_payload(paths, stage, args.profile, system, exe_suffix, lib_suffix)
    versions = {"cyxwiz": version}

    if args.profile == "full":
        assert arrayfire_root is not None
        assert python_root is not None
        assert full_python_version is not None
        assert intel_notices is not None
        versions["arrayfire"] = package_arrayfire(arrayfire_root, stage, backends, lib_suffix)
        versions["python"] = full_python_version
        copy_tree(python_root, stage / "python")
        copy_runtime_notices(stage, intel_notices, nvidia_notices)

    create_launcher(stage, args.profile, system)
    backend_text = ", ".join(backends) if backends else "externally installed selection"
    render_readme(
        paths.templates / ("README_FULL.md" if args.profile == "full" else "README_MINIMAL.md"),
        stage / "README.md",
        {
            "VERSION": version,
            "PLATFORM": platform_name,
            "BACKENDS": backend_text,
            "ARRAYFIRE_VERSION": versions.get("arrayfire", "external; see compatibility requirements"),
        },
    )
    write_manifest(stage, args.profile, version, platform_name, backends, versions)

    if args.stage_only:
        return stage, None
    backend_suffix = "" if args.profile == "minimal" else "-" + "-".join(backends)
    name = f"cyxwiz-engine-{version}-{args.profile}{backend_suffix}-{platform_name}"
    archive = create_archive(stage, paths.output_root, name, system)
    return stage, archive


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        stage, archive = build_package(args)
    except PackageError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    print(f"[OK] Staged package: {stage}")
    if archive:
        print(f"[OK] Archive: {archive} ({archive.stat().st_size} bytes)")
    else:
        print("[OK] Archive creation skipped (--stage-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
