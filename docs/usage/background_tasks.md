# Background Tasks

Long work runs off the screen's thread so the Engine stays responsive:
loading a dataset, profiling, sorting a big table, building a cache, a
training launch. **Background Tasks** lists that work while it runs and keeps
the last tasks that finished.

Open it from the sidebar (hourglass), **View > Panels > Training > Background
Tasks**, or click **N tasks running** at the right of the status bar. Other
windows that start such work say "Background Tasks shows it".

## What it shows

- **The summary** at the top: how many tasks run and how many finished in
  this session, e.g. *1 running, 2 finished*.
- **The last training run**, in one line: its outcome (*finished*, *stopped
  early*, *cancelled*, *failed*, or *Training now*), where it ran (*on CUDA,
  NVIDIA GeForce GTX 1050 Ti*) and when it ended. **Open in Training
  Dashboard** shows the run itself; its placement, residency, transfers and
  synchronisations are under Run details > Where it ran there.
- **Running**: the tasks in the order they started. Each shows its name, its
  latest message, a progress bar with the percentage and the time so far, and
  a cancel button when the task can stop. A task that waits for a worker
  says *waiting to start*. With nothing running the list says so.
- **Finished**: the last 20, newest first, each with its message, how long it
  took (*took 12.6 s*, *cancelled after 2.5 s*) and the time it ended. A
  failed task says why in the failure colour. When the Engine holds more than
  20 (it keeps the last 100), the heading says *last 20 of 57*.

## Controls

| Control | What it does |
| --- | --- |
| Show finished | Shows or hides the finished list for this session. Hidden, the window says how many are hidden. Your default is in **Preferences > Tools > Background Tasks**. |
| Cancel all | Asks every task that can stop to stop. Off, with the reason on hover, when nothing running can be cancelled. Cancelling is cooperative: a task stops at its next check. |
| Cancel (x) on a row | Cancels that one task. |
| Open in Training Dashboard | Opens the Training Dashboard on the last run. |

The status bar shows a spinner and *N tasks running* while anything runs;
hover it for each task's progress, click it to open Background Tasks.
