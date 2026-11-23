"""The Monitor window's system figures against psutil.

    python compare_system_metrics.py <path to test_system_metrics>

Starts `test_system_metrics --print`, which prints one reading and stays alive,
then asks psutil about the machine and about that process. Exit code 0 when
they agree, 77 when psutil is not installed (ctest reports a skip).
"""
import json
import os
import subprocess
import sys

try:
    import psutil
except ImportError:
    print("psutil is not installed; nothing compared")
    sys.exit(77)

failures = []


def check(ok, what):
    print(("ok      " if ok else "FAILED  ") + what)
    if not ok:
        failures.append(what)


def within(a, b, share):
    return abs(a - b) <= share * max(a, b, 1)


child = subprocess.Popen([os.path.abspath(sys.argv[1]), "--print"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
try:
    ours = json.loads(child.stdout.readline())
    process = psutil.Process(ours["pid"])
    memory = psutil.virtual_memory()
    rss = process.memory_info().rss
    threads = process.num_threads()

    check(ours["cores"] == psutil.cpu_count(logical=True), f'cores {ours["cores"]} vs psutil {psutil.cpu_count(logical=True)}')
    check(ours["memory_known"], "memory is known")
    check(within(ours["memory_total"], memory.total, 0.001), f'memory total {ours["memory_total"]} vs psutil {memory.total}')
    # Two readings a moment apart on a working machine.
    check(within(ours["memory_available"], memory.available, 0.05) or abs(ours["memory_available"] - memory.available) < (512 << 20),
          f'memory available {ours["memory_available"]} vs psutil {memory.available}')
    check(ours["process_memory_known"] and within(ours["process_memory"], rss, 0.10),
          f'process memory {ours["process_memory"]} vs psutil rss {rss}')
    check(ours["threads_known"] and ours["threads"] == threads, f'threads {ours["threads"]} vs psutil {threads}')
    check(ours["cpu_known"] and 0.0 <= ours["cpu_percent"] <= 100.0, f'machine CPU {ours["cpu_percent"]:.1f} % is 0..100')
finally:
    try:
        child.stdin.write("\n")
        child.stdin.close()
    except OSError:
        pass
    child.wait(timeout=10)

if failures:
    print(f"{len(failures)} figure(s) disagree with psutil")
    sys.exit(1)
print("system metrics agree with psutil")
