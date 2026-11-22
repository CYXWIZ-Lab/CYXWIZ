#!/usr/bin/env python3
"""The oracle nightly for one lab machine (TOFIX228 1.7, owner P2: a local
scheduled script, no self-hosted runner).

In a clone of its own (never the working tree someone is using), it:
  1. fetches and checks out the branch's latest pushed commit,
  2. configures a build with this machine's devices (CYXWIZ_ORACLE_DEVICES),
  3. builds only the oracle tests (target cyxwiz-oracle-tests),
  4. runs `ctest -L oracle` (every registered device),
  5. with --live, re-runs every fixture generator with the PyTorch / sklearn /
     scipy installed here and reports which committed fixtures no longer
     reproduce, and by how much (the live half of D10),
and writes a dated report folder: report.md, the ctest JUnit file and logs.
The exit code is 0 only when every step passed.

Example (Dev PC, CUDA + OpenCL, Windows):
  py -3.12 scripts/oracle/nightly_oracle.py --clone D:/oracle-nightly/CyxWiz
      --reports D:/oracle-nightly/reports --devices "cpu;cuda;opencl" --live
      --cmake-arg=-DCMAKE_TOOLCHAIN_FILE=D:/Dev/CyxWiz_Engine/vcpkg/scripts/buildsystems/vcpkg.cmake
      --cmake-arg=-DVCPKG_INSTALLED_DIR=D:/Dev/CyxWiz_Engine/vcpkg_installed
      --cmake-arg=-DVCPKG_MANIFEST_INSTALL=OFF
Scheduling: see scripts/oracle/README.md (Task Scheduler / cron lines).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_REMOTE = "https://github.com/CYXWIZ-Lab/CYXWIZ.git"
FIXTURES = "cyxwiz-engine/tests/computation_truth/fixtures"
REFERENCE = "cyxwiz-engine/tests/computation_truth/reference"


class Report:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.lines: list[str] = []
        self.failed = False

    def step(self, name: str, ok: bool, detail: str = "") -> None:
        self.failed |= not ok
        self.lines.append(f"- **{name}**: {'pass' if ok else 'FAIL'}{' - ' + detail if detail else ''}")
        print(f"[{'pass' if ok else 'FAIL'}] {name} {detail}", flush=True)

    def write(self, header: list[str]) -> None:
        text = "\n".join(header + [""] + self.lines) + "\n"
        (self.folder / "report.md").write_text(text, encoding="utf-8")


def run(command: list[str], cwd: Path, log: Path, env: dict | None = None, timeout: int | None = None) -> int:
    with log.open("ab") as out:
        out.write(("\n$ " + " ".join(command) + "\n").encode())
        out.flush()
        try:
            return subprocess.run(command, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, env=env,
                                  timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            out.write(b"\n(timed out)\n")
            return 124


def numeric_difference(old: object, new: object) -> tuple[float, str]:
    """The largest |old - new| between two JSON values of the same shape, and
    where it is ("" when equal); inf where the structure differs."""
    worst, where = 0.0, ""
    stack = [(old, new, "")]
    while stack:
        a, b, path = stack.pop()
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                return float("inf"), path or "/"
            stack.extend((a[k], b[k], f"{path}/{k}") for k in a)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return float("inf"), path or "/"
            stack.extend((x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b)))
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            diff = abs(float(a) - float(b))
            if diff > worst:
                worst, where = diff, path
        elif a != b and not path.endswith(("version", "torch_version")):
            return float("inf"), path or "/"
    return worst, where


def live_fixtures(clone: Path, python: str, report: Report, logs: Path) -> None:
    """Re-run every generator in the clone; compare each rewritten fixture with
    the committed one."""
    manifest = json.loads((clone / FIXTURES / "fixture_manifest.json").read_text(encoding="utf-8"))["fixtures"]
    generators = sorted({entry["generator"] for name, entry in manifest.items() if "imagenet" not in name})
    failed_generators = []
    for generator in generators:
        script = clone / "cyxwiz-engine/tests/computation_truth" / generator
        if run([python, script.name], script.parent, logs / "live_generators.log", timeout=3600) != 0:
            failed_generators.append(generator)
    report.step("live: every generator runs", not failed_generators,
                f"{len(generators) - len(failed_generators)} of {len(generators)}"
                + (f"; failed: {', '.join(failed_generators)}" if failed_generators else ""))
    changed = subprocess.run(["git", "diff", "--name-only", "--", FIXTURES], cwd=clone,
                             capture_output=True, text=True).stdout.split()
    rows = []
    for path in changed:
        name = Path(path).name
        if not name.endswith(".json"):
            rows.append(f"  - {name}: binary or graph file changed")
            continue
        committed = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=clone, capture_output=True,
                                   text=True, encoding="utf-8").stdout
        diff, where = numeric_difference(json.loads(committed), json.loads((clone / path).read_text(encoding="utf-8")))
        rows.append(f"  - {name}: largest difference {diff:.3g} at {where}")
    report.step("live: committed fixtures reproduce", not changed,
                f"{len(changed)} fixture(s) differ" if changed else "all equal")
    report.lines.extend(rows)
    subprocess.run(["git", "checkout", "--", FIXTURES], cwd=clone)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clone", type=Path, required=True, help="a clone used only by this script")
    parser.add_argument("--reports", type=Path, required=True, help="where dated report folders go")
    parser.add_argument("--remote", default=DEFAULT_REMOTE)
    parser.add_argument("--branch", default="Nodes_Implementation")
    parser.add_argument("--devices", default="cpu", help='e.g. "cpu;cuda;opencl" (CYXWIZ_ORACLE_DEVICES)')
    parser.add_argument("--build-dir", default="build-oracle", help="relative to the clone")
    parser.add_argument("--cmake-arg", action="append", default=[], help="extra configure argument (repeat)")
    parser.add_argument("--python", default=sys.executable, help="the Python with torch for --live")
    parser.add_argument("--live", action="store_true", help="re-run the fixture generators (needs torch etc.)")
    parser.add_argument("--parallel", default="4", help="build and ctest parallelism")
    args = parser.parse_args()

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    folder = args.reports / f"{stamp}_{platform.node()}"
    logs = folder / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    report = Report(folder)
    clone = args.clone.resolve()

    if not (clone / ".git").exists():
        clone.parent.mkdir(parents=True, exist_ok=True)
        ok = run(["git", "clone", args.remote, str(clone)], clone.parent, logs / "git.log") == 0
        report.step("clone", ok)
        if not ok:
            report.write([f"# Oracle nightly {stamp} on {platform.node()}"])
            return 1
    ok = (run(["git", "fetch", "origin", args.branch], clone, logs / "git.log") == 0 and
          run(["git", "checkout", "--force", "--detach", f"origin/{args.branch}"], clone, logs / "git.log") == 0 and
          run(["git", "clean", "-fdx", "-e", args.build_dir], clone, logs / "git.log") == 0)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone, capture_output=True, text=True).stdout.strip()
    report.step("fetch", ok, f"origin/{args.branch} at {commit[:12]}")
    header = [f"# Oracle nightly {stamp} on {platform.node()}",
              "", f"- commit: {commit}", f"- devices: {args.devices}", f"- OS: {platform.platform()}"]

    build = clone / args.build_dir
    if ok:
        configure = ["cmake", "-S", ".", "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                     "-DCYXWIZ_BUILD_TESTS=ON", f"-DCYXWIZ_ORACLE_DEVICES={args.devices}"] + args.cmake_arg
        ok = run(configure, clone, logs / "configure.log") == 0
        report.step("configure", ok)
    if ok:
        ok = run(["cmake", "--build", str(build), "--config", "Release", "--target", "cyxwiz-oracle-tests",
                  "--parallel", args.parallel], clone, logs / "build.log") == 0
        report.step("build the oracle tests", ok)
    if ok:
        junit = folder / "oracle-junit.xml"
        code = run(["ctest", "--test-dir", str(build), "--build-config", "Release", "-L", "oracle",
                    "--output-on-failure", "--timeout", "1800", "--parallel", args.parallel,
                    "--output-junit", str(junit)], clone, logs / "ctest.log")
        summary = ""
        for line in (logs / "ctest.log").read_text(encoding="utf-8", errors="replace").splitlines():
            if "tests passed" in line:
                summary = line.strip()
        report.step("ctest -L oracle", code == 0, summary)
        last = build / "Testing" / "Temporary" / "LastTest.log"
        if last.exists():
            shutil.copy2(last, logs / "LastTest.log")
    if args.live and (clone / FIXTURES).exists():
        live_fixtures(clone, args.python, report, logs)

    report.write(header)
    print(f"report: {folder / 'report.md'}")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
