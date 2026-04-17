#!/usr/bin/env bash
# scripts/rebuild.sh - safe incremental rebuild wrapper.
#
# Why this exists (2026-04-17 lesson):
#   Bash pipelines return the LAST command's exit code, not the first's.
#   So "cmake --build ... | tail -3" silently reports success even when
#   cmake fails - tail always exits 0. We wasted ~1h running a zombie
#   binary because a real C2440 compile error was masked this way, and
#   the incremental-build engine happily kept the prior .exe on disk so
#   manual testing looked like it was exercising the new code.
#
# This wrapper:
#   1. Captures build output to a file (no pipe that masks exit code).
#   2. Checks cmake's real exit code.
#   3. Verifies the binary's mtime actually moved forward (catches the
#      case where MSBuild "succeeded" but didn't regenerate artifacts).
#   4. Surfaces warnings without drowning in noise.
#
# For a fresh-setup build with vcpkg/ninja, use scripts/build.sh. This
# wrapper is for the iterative edit-compile-test loop.
#
# Usage:
#   scripts/rebuild.sh                 # default target: cyxwiz-engine
#   scripts/rebuild.sh <target>        # override target
#
# Env:
#   PRESET=Release (default) | Debug
#   BUILD_DIR=build (default)

set -u

TARGET="${1:-cyxwiz-engine}"
PRESET="${PRESET:-Release}"
BUILD_DIR="${BUILD_DIR:-build}"
BINARY="${BUILD_DIR}/bin/${PRESET}/cyxwiz-engine.exe"
LOG="${BUILD_DIR}/last-build.log"

echo "==> Building ${TARGET} (${PRESET})"

# Record binary mtime before build. `stat -c %Y` gives unix epoch seconds
# (git-bash on Windows supports it via MSYS). Missing binary = 0.
mtime_before=$(stat -c %Y "${BINARY}" 2>/dev/null || echo 0)

# Capture to file so we can examine without a pipe masking exit code.
cmake --build "${BUILD_DIR}" --config "${PRESET}" --target "${TARGET}" > "${LOG}" 2>&1
rc=$?

if [ ${rc} -ne 0 ]; then
    echo "==> BUILD FAILED (cmake exit ${rc})"
    echo "Last 40 lines of ${LOG}:"
    echo "---"
    tail -40 "${LOG}"
    exit 1
fi

# MSBuild can occasionally report success while not actually producing
# a fresh .exe (e.g. when a skipped TU that would have failed is cached
# as up-to-date). If the binary didn't move forward, flag it loudly.
mtime_after=$(stat -c %Y "${BINARY}" 2>/dev/null || echo 0)
if [ "${mtime_after}" -le "${mtime_before}" ] && [ "${mtime_before}" -gt 0 ]; then
    echo "==> WARNING: cmake OK but binary mtime unchanged."
    echo "    Either nothing needed rebuilding, or MSBuild skipped a"
    echo "    failing TU and reused stale .obj. Review ${LOG} to confirm."
fi

# Warning summary. Unique MSVC warnings, first 10 lines.
warn_count=$(grep -c "warning C[0-9]*:" "${LOG}" 2>/dev/null || echo 0)
# grep -c prints the count. But if grep returns nonzero (no matches),
# the variable may contain an error message on some systems; force to 0.
case "${warn_count}" in
    ''|*[!0-9]*) warn_count=0 ;;
esac

echo "==> Warnings: ${warn_count}"
if [ "${warn_count}" -gt 0 ]; then
    grep -E "warning C[0-9]*:" "${LOG}" | sort -u | head -10
    if [ "${warn_count}" -gt 10 ]; then
        echo "    ... (${warn_count} total lines; see ${LOG} for full list)"
    fi
fi

if [ -f "${BINARY}" ]; then
    # Print size + mtime in a single line.
    ls -la "${BINARY}" | awk '{print "==> Binary: " $5 " bytes  " $6 " " $7 " " $8}'
fi

echo "==> OK"
