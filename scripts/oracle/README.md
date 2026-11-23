# Oracle nightly (lab machines)

`nightly_oracle.py` runs the oracle tests on one machine's devices, in a
clone of its own, and writes a dated report. It is the nightly half of the
oracle harness (TOFIX228 1.7, owner decision P2): a local scheduled script on
each lab machine, not a GitHub self-hosted runner (a public repository's
runner would run code from any pull request on the owner's machines). The
per-push half is `.github/workflows/oracle-cpu.yml` (ArrayFire CPU).

What one run does:

1. Fetches `origin/<branch>` (default `Nodes_Implementation`) into the clone
   given by `--clone` and checks it out (pushed code only; never point
   `--clone` at a working tree in use: it is reset and cleaned).
2. Configures with `CYXWIZ_ORACLE_DEVICES` set to `--devices`.
3. Builds only the target `cyxwiz-oracle-tests`.
4. Runs `ctest -L oracle` on every registered device.
5. With `--live`: re-runs every fixture generator with the PyTorch /
   scikit-learn / scipy installed here and lists the committed fixtures that
   no longer reproduce, with the largest numeric difference.

The live step needs the generators' packages: `py -3.12 -m pip install -r
scripts/oracle/requirements-live.txt`. Fixtures whose generator needs data that
is not committed (the penguins CSV, the ImageNet weights) carry `live_skip` in
`fixture_manifest.json` and are listed, not run.

Report: `<reports>/<date>_<time>_<machine>/report.md`, `oracle-junit.xml` and
`logs/` (configure, build, ctest, LastTest.log, live generators). Exit code 0
only when every step passed.

## One machine

| Machine | Devices | Notes |
|---------|---------|-------|
| Dev PC | `cpu;cuda;opencl` | GTX 1050 Ti (CUDA, OpenCL), Intel UHD (OpenCL 1) |
| Dell | `cpu;oneapi` | Intel Iris Xe: float32 only, oneAPI |
| Mac Intel / Linux hosts | `cpu` (+ `opencl` where present) | |

Example (Dev PC):

```
py -3.12 scripts/oracle/nightly_oracle.py --clone D:/oracle-nightly/CyxWiz ^
  --reports D:/oracle-nightly/reports --devices "cpu;cuda;opencl" --live ^
  --cmake-arg=-DCMAKE_TOOLCHAIN_FILE=D:/Dev/CyxWiz_Engine/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  --cmake-arg=-DVCPKG_INSTALLED_DIR=D:/Dev/CyxWiz_Engine/vcpkg_installed ^
  --cmake-arg=-DVCPKG_MANIFEST_INSTALL=OFF
```

Reusing an existing `vcpkg_installed` folder avoids building the dependencies
again; the first run of a new clone still compiles the Engine's oracle targets.

## Scheduling (not set up by the script)

Windows, Task Scheduler, every night at 02:00 (put the command above in
`D:\oracle-nightly\run.cmd` first):

```
schtasks /Create /TN "CyxWiz oracle nightly" /SC DAILY /ST 02:00 /TR "D:\oracle-nightly\run.cmd"
```

Linux / macOS, cron (`crontab -e`):

```
0 2 * * * cd $HOME/oracle-nightly && python3 CyxWiz/scripts/oracle/nightly_oracle.py --clone $HOME/oracle-nightly/CyxWiz --reports $HOME/oracle-nightly/reports --devices cpu >> nightly.log 2>&1
```

The GPU entries take a lock per device (ctest `RESOURCE_LOCK`), and a training
run holds its GPU's lease (`GpuRunLease`), so the nightly waits for, rather
than collides with, another training run on the same GPU.
