# Compiler warning gate

This tool classifies warnings captured from a normal CMake build without
changing compiler flags or suppressing diagnostics. Warnings are separated
into these ownership groups:

- `owned`: CyxWiz source; enforced against the committed zero-warning limit.
- `vendored`: third-party source kept inside a CyxWiz source directory.
- `external`: package-manager, SDK, and system headers.
- `generated`: build-tree and generated dependency source.
- `toolchain`: linker and build-tool diagnostics without a source location.

The gate intentionally applies only to `owned` diagnostics. Every warning is
still retained in the JSON artifact for dependency and link-graph follow-up.

## Reproduce on macOS or Linux

Use a clean build so every translation unit is compiled:

```sh
mkdir -p warning-artifacts
set -o pipefail
cmake --build build --clean-first --parallel 2>&1 | tee warning-artifacts/build.log
python3 scripts/compiler_warnings/report.py \
  --log warning-artifacts/build.log \
  --platform macos \
  --baseline scripts/compiler_warnings/baseline.json \
  --repo-root . \
  --output-dir warning-artifacts
```

Use `--platform linux` for a Linux build.

## Reproduce on Windows PowerShell

```powershell
New-Item -ItemType Directory -Force warning-artifacts | Out-Null
cmake --build build --config Release --clean-first --parallel 2>&1 |
  Tee-Object -FilePath warning-artifacts/build.log
python scripts/compiler_warnings/report.py `
  --log warning-artifacts/build.log `
  --platform windows `
  --baseline scripts/compiler_warnings/baseline.json `
  --repo-root . `
  --output-dir warning-artifacts
```

The command emits `warning-report.json` for automation and
`warning-report.md` for human review. A non-zero exit status means the owned
warning limit increased or the input/baseline could not be read.
