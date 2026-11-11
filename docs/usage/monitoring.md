# Monitoring and diagnostics

**Tools > Monitoring** opens one **Monitor** window with four tabs. Each menu
item opens it on its own tab:

| Tab | What it shows | Menu item |
| --- | --- | --- |
| System | The machine, the compute device and this Engine: CPU, memory, threads, with trends | System |
| Memory | The Engine's memory over time, and now / lowest / highest for each kind | Memory |
| Loaded data | Every dataset the Engine holds, in memory and on disk (see [Loaded data](loaded_data.md)) | Loaded Data |
| Frames | Frame times and the windows they go to | Frames |

The same menu has **Performance Profiler** (opens Studio Debugger > Runtime,
where a training run spends its time), **Clear Caches...** and **Free Unused
Memory**. **Tools > Diagnostics** holds three switches, described
[below](#diagnostics).

Every figure is measured. A figure the system does not give reads **Not
available**, and a device whose driver does not report its free memory reads
**Not reported by this driver**. Nothing is estimated or filled in.

## How the readings are taken

A reader on its own thread takes one reading every **Sample every** seconds
(1 by default, 0.25 to 60) and keeps them for **Keep** (5 minutes, 15 minutes
or 1 hour). It runs for as long as the Engine does, also while the window is
closed, so the history is already there when you open it.

**Save as my default** keeps the tab, the interval and the history length for
next time. The same three settings are in **Preferences > General > Tools >
Monitoring and Diagnostics**.

## System

Four cards:

- **CPU**: the share of all cores in use, and the number of cores.
- **Memory**: used and total main memory, and how much programs can still
  take ("free" counts memory the system would give up, as Task Manager's
  "available" does).
- **Compute device**: the device the Engine computes on, in three lines.
  - **Held by the Engine**: what the Engine's memory manager has allocated on
    the device, and how much of that is in use. The rest is kept for reuse.
  - **Device memory**: free and total for every program, as the driver
    reports it. NVIDIA cards answer through the NVIDIA management library
    (what `nvidia-smi` reads). Other drivers do not report it.
  - **Other programs**: the device's used memory less what the Engine holds
    (browsers, other training jobs, the desktop).
- **This Engine**: the process's memory (its working set), its share of the
  machine's CPU, and its threads.

The meters turn amber from 70 % and red from 90 %.

With no graphics device in use the card reads **CPU only**.

### Free unused memory

Returns the device memory the Engine keeps for reuse to the driver, asks
Python to collect its garbage, and says what happened, for example:

> Freed 412 MB of device memory the Engine was holding but not using. Python
> collected 96 objects.

Memory in use stays: this does not make a running job smaller. It helps when
another program needs the device memory the Engine is only keeping.

### Clear caches

Lists every cache with its size and what it holds; tick the ones to clear.
It opens at any time, also before the first training run. It is the same
set as **Preferences > Storage**: the temporary Parquet copies of large
files on this PC and, with a project open, the project's prepared data and
query results. Files that loaded data still reads are kept.

## Memory

One chart of five series, each with a checkbox:

| Series | What it is | Measured by |
| --- | --- | --- |
| Engine process | The whole process in main memory | the operating system |
| Device, allocated | Held on the compute device (in use plus kept for reuse) | ArrayFire's memory manager |
| Device, in use | The part of that in use by live arrays | ArrayFire's memory manager |
| Tables (Arrow) | Loaded tables | Arrow's memory pool |
| Tensors on the host | Tensors in main memory | the tensor allocator |

The table under it gives **Now**, **Lowest** and **Highest** over the kept
history. A reading that did not have a figure is left out of the chart and
the table; it is not drawn as zero.

**Export CSV...** asks where to save and writes one line per reading (an
unknown figure is an empty cell). **Clear history** starts the history again.

While a training run owns the device, its own thread reports the device
figures, so they stay live during the run.

## Frames

Two times, kept apart:

- **Whole frame**: what the screen shows. It includes waiting for the display
  and the pauses the Engine takes when idle, so 10 or 30 frames a second here
  does not mean the Engine is slow.
- **Building the interface**: the Engine's own work each frame, now, on
  average and at worst over the last 180 frames.

The table lists the windows drawn in this frame with their time, as measured,
and their running average. **Frame rate now** says which pace the Engine is
in: full speed, 30 frames a second (another window is in front), 10 frames a
second (idle: no input for a second) or not drawing (minimised). Training
keeps a focused Engine at full speed.

Frames are recorded only while this tab is open or the overlay is on, and
the history starts again each time.

## Diagnostics

Three switches in **Tools > Diagnostics**. They are saved, so they are still
on after a restart, and are also in **Preferences > General**.

| Switch | What it does |
| --- | --- |
| Log Frame-rate Changes | One log line each time the pace changes, e.g. `Frame rate: 10 frames a second (idle: no input for a second)` |
| Log Every Python Command | One log line per command, script or run sent to Python, when it ends, with its duration: `Python command took 3.2 ms: x = 1 + 1` |
| Frame Time Overlay | A small panel at the top right with the two frame times and the three costliest windows; its link opens Monitor > Frames |

## Python

`pycyxwiz.system` gives the Engine's memory figures to scripts:

```python
import pycyxwiz as cx
s = cx.system

cx.initialize()
s.device_memory_held()            # {'known': True, 'allocated_bytes': 0, 'in_use_bytes': 0, 'buffers': 0, 'buffers_in_use': 0}
batch = cx.Tensor.random([4096, 1024], seed=1)
s.device_memory_held()            # allocated 16 MB, in use 16 MB, 1 buffer
del batch
s.device_memory_held()            # allocated 16 MB, in use 0: kept for reuse
s.release_unused_device_memory()  # {'ran': True, 'allocated_before': 16777216, 'allocated_after': 0, 'freed_bytes': 16777216}
s.device_memory_free()            # {'known': True, 'total_bytes': 4294967296, 'free_bytes': ..., 'source': 'NVIDIA management library'}
s.host_tensor_bytes(), s.host_tensor_peak_bytes()
s.reset_host_tensor_peak()
```

`device_memory_free(hardware_uuid="")` takes a device's
`DeviceInfo.hardware_uuid`; without one it answers for the machine's only
card. When the driver does not report, `known` is False and `source` says
"Not reported by this driver". For the machine's CPU and main memory use
`psutil`; the Monitor window's own figures agree with it.

The example `examples/python/system_memory.py` runs these and prints one
line per step.

## C++ (plugin SDK)

`cyxwiz/device_memory.h`: `ReadDeviceMemoryHeld`, `LastDeviceMemoryHeld`
(the last reading, for a watcher thread: no device call),
`ReadDeviceMemoryFree` and `ReleaseUnusedDeviceMemory`, which now answers
what it gave back.
