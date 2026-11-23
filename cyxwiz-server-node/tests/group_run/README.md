# A whole training group on one machine

These two scripts run a group job end to end on one PC: the coordinator
(`cyxwiz-coordinator`, from the central server's repository) and several
group workers (`group_worker_tool`, a Server Node's group worker without the
rest of the daemon). Workers train on the ArrayFire CPU route, so no verified
GPU is needed. They are how the group path is checked before it goes onto
separate machines.

## Build

- `group_worker_tool` builds with the node's tests (`CYXWIZ_BUILD_TESTS`), into
  `build/bin/Release`.
- The coordinator: see the central server's `docs/usage/coordinator.md`.

## Make a job

```
python make_job.py D:\jobs\one-pc 2
```

This writes, into the folder:

- 200,000 rows of a three-class problem, shuffled, in four Parquet parts.
- A small classifier graph (`model.cyxgraph`).
- `job.json`, set for 2-second rounds and one epoch.
- One ticket file per worker.

Needs `numpy` and `pyarrow`.

## Run it

```
.\run_local.ps1 -JobDir D:\jobs\one-pc -Coordinator <cyxwiz-coordinator.exe> -Tool <group_worker_tool.exe> -Scenario plain
```

| Scenario | What happens | What should be seen |
| --- | --- | --- |
| `plain` | Every worker runs to the end of the job. | `completed: True`; every worker `Group finished: completed`; no update refused. |
| `drop` | The last worker is killed after `-AfterSeconds`. | The coordinator removes it after two missed rounds; the others finish. |
| `restart` | The coordinator is killed after `-AfterSeconds` and started again. | `going on from the stored state round=N`; the workers join again and finish; each has one update refused (the round that was open). |

The script prints the rounds committed, the coordinator's notable lines, and
for each worker the steps per round (first, median, last) and its last line.
The logs are left in the job folder (`coordinator*.log`, `worker*.log`).

To see a job whose training diverges, copy the folder, set the Adam node's
`learning_rate` to `1e30` in `model.cyxgraph`, and give `job.json` another
`state_dir`. The workers stop with "training produced a value that is not a
finite number", and the coordinator ends the job as not started.

## What these runs found (2026-11-23)

- **A slowdown:** steps per round fell to a third after about ten rounds. It
  was a leak in the training health monitor; fixed in the Engine.
- **A coordinator that waited for good** when every worker left before
  round 1. Fixed in the coordinator.
- **About 1 s of fixed cost per round on a worker:** batchers and device
  preflight 0.35 s, per-run setup 0.33 s, teardown 0.21 s. That is about 3 % of
  a 30-second round.
