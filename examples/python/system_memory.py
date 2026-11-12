"""pycyxwiz.system: the Engine's memory from Python.

Nothing to download. Shows what the process holds on its compute device before
and after it makes a tensor, what releasing unused memory gives back, what the
device itself has free, and the host tensor counters: the figures of Tools >
Monitoring > System and Memory.

    python system_memory.py <folder holding pycyxwiz> [<folder of its DLLs>]

The numbers depend on the machine; python_doc/pycyxwiz.md (6.7) and
docs/usage/monitoring.md show one run.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cyxwiz-backend", "python", "tests"))
from pycyxwiz_test_env import import_pycyxwiz  # noqa: E402

cyx = import_pycyxwiz()
system = cyx.system


def size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024


def held(label):
    h = system.device_memory_held()
    print(f"{label:<28} allocated {size(h['allocated_bytes']):>10}   in use {size(h['in_use_bytes']):>10}   "
          f"buffers {h['buffers']} ({h['buffers_in_use']} in use)")


print("before initialize():", system.device_memory_held()["known"])  # False: no device yet
cyx.initialize()

held("after initialize()")
batch = cyx.Tensor.random([4096, 1024], seed=1)  # 16 MB of float32
held("with a 16 MB tensor")
del batch
held("tensor dropped")  # still allocated: the buffer is kept for reuse

released = system.release_unused_device_memory()
print(f"released                     {size(released['freed_bytes'])} "
      f"({size(released['allocated_before'])} -> {size(released['allocated_after'])})")
held("after the release")

# The device itself, for every program. Only some drivers answer.
free = system.device_memory_free()
if free["known"]:
    print(f"device                       {size(free['free_bytes'])} free of {size(free['total_bytes'])} ({free['source']})")
else:
    print(f"device                       {free['source']}")

# Host tensors: main memory held by tensors that are not on the device.
print(f"host tensors                 {size(system.host_tensor_bytes())} now, peak {size(system.host_tensor_peak_bytes())}")
system.reset_host_tensor_peak()
print(f"peak after reset             {size(system.host_tensor_peak_bytes())}")
