# SPDX-License-Identifier: Apache-2.0
# Part of the CyxWiz plugin SDK examples (see cmake/sdk/NOTICE).
"""pycyxwiz.system: the Engine's memory figures.

What the process holds on its compute device follows a tensor that is made and
dropped; releasing unused device memory reports what it gave back; the device's
own free memory is known or says who did not answer; the host tensor counters
move with a host tensor. Run with the Python pycyxwiz was built for:

    python test_python_system.py <folder holding pycyxwiz> [<folder of its DLLs>]

Exit code 0 when every check passes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pycyxwiz_test_env import Checks, import_pycyxwiz  # noqa: E402

cyx = import_pycyxwiz()
check = Checks()
s = cyx.system

cyx.initialize()

# ---- What the process holds on its device
before = s.device_memory_held()
check(before["known"], "held memory is known after initialize()")
check(set(before) == {"known", "allocated_bytes", "in_use_bytes", "buffers", "buffers_in_use"}, "the keys of device_memory_held")
check(before["allocated_bytes"] >= before["in_use_bytes"], "allocated covers in use")

block = cyx.Tensor.random([1024, 1024], seed=1)  # 4 MB of float32, drawn on the device
during = s.device_memory_held()
check(during["in_use_bytes"] >= before["in_use_bytes"] + 4 * 1024 * 1024,
      f'a 4 MB tensor is in use: {before["in_use_bytes"]} -> {during["in_use_bytes"]}')
check(during["buffers_in_use"] >= before["buffers_in_use"] + 1, "one more buffer in use")
del block
after = s.device_memory_held()
check(after["in_use_bytes"] < during["in_use_bytes"], "dropping the tensor ends its use")

# ---- Release: the buffer kept for reuse goes back to the driver
released = s.release_unused_device_memory()
check(released["ran"], "release ran")
check(released["freed_bytes"] == released["allocated_before"] - released["allocated_after"], "freed = before - after")
check(released["freed_bytes"] >= 4 * 1024 * 1024, f'the 4 MB kept for reuse was freed: {released["freed_bytes"]}')
check(s.device_memory_held()["allocated_bytes"] == released["allocated_after"], "held matches what release reported")
check(s.release_unused_device_memory()["freed_bytes"] == 0, "nothing left to free")

# ---- The device's own free memory: known, or it says who did not answer
nobody = s.device_memory_free("00000000000000000000000000000000")
check(not nobody["known"] and nobody["total_bytes"] == 0 and nobody["source"] == "Not reported by this driver",
      "an unknown card is not guessed")
only = s.device_memory_free()
if only["known"]:
    check(0 < only["total_bytes"] and only["free_bytes"] <= only["total_bytes"], "free is within total")
    check(only["source"] == "NVIDIA management library", "the source is named")
    for device in cyx.get_available_devices():
        if device.hardware_uuid_known and s.device_memory_free(device.hardware_uuid)["known"]:
            check(device.memory_available_known and device.memory_available <= device.memory_total,
                  f"{device.name}: the listed device carries its free memory")
else:
    check(only["source"] == "Not reported by this driver", "no answer is said so")

# ---- Host tensors
check(s.host_tensor_peak_bytes() >= s.host_tensor_bytes(), "the peak is at least the current value")
s.reset_host_tensor_peak()
check(s.host_tensor_peak_bytes() == s.host_tensor_bytes(), "reset starts the peak from the current value")

sys.exit(check.finish("pycyxwiz.system reports the Engine's memory"))
