---
title: Crash dumps and logs
description: Where the Engine writes its logs and run records, how to record Debug messages, where Windows keeps crash reports, and what to attach to a bug report.
type: troubleshooting
area: general
audience: user
status: stable
since: 1.0.7
owner: general
last_reviewed: 2026-11-25
related:
  - index.md
  - ../main-window/console.md
  - ../user-guide/training/studio-debugger.md
  - ../main-window/preferences.md
---

When the Engine closes unexpectedly or a run fails without a clear message,
the logs and the last run record show what happened. Collect them before
you start the Engine many times: it keeps only the last five logs.

## The Engine closed unexpectedly

**Cause:** an error the Engine could not recover from, in the Engine, a GPU
runtime, a plugin or a Python node.

**Fix:**

1. Start the Engine again. The log of the launch that crashed is now
   `engine_log.1.txt`.
2. Choose **Help > Open Logs Folder** and open `engine_log.1.txt`. Read the
   last **error** and **critical** lines. An exception nobody caught ends
   with "Unhandled exception, the Engine must close: ...".
3. If the crash happened during training, open the **Studio Debugger**:
   its **Runtime Trace** and **Runtime > Crash** views read the last run
   record and, on Windows, the matching Windows crash report.
4. Check [Known issues](../known-issues.md) for the device you trained on.
5. Report it with the files in
   [What to attach to a bug report](#what-to-attach-to-a-bug-report).

A plugin that crashes inside one of its calls is caught: the Console says
"Plugin '...' crashed during ...: Access Violation" and the plugin
is stopped, not the Engine. See [Plugins and Python](plugins-and-python.md).

## Where the logs are

The Engine writes every message to `engine_log.txt` in the folder of the
Engine program (`cyxwiz-engine`), and shows the same messages in the
**Console**.

| File | Holds |
| --- | --- |
| `engine_log.txt` | This launch |
| `engine_log.1.txt` | The launch before (the crashed one, after a restart) |
| `engine_log.2.txt` to `engine_log.4.txt` | Older launches; the oldest is removed at each start |

**Help > Open Logs Folder** opens the folder in the file manager. The
Console's levels, search and copy commands are described in
[Console and logs](../main-window/console.md).

## The log has no Debug lines

**Cause:** **Record Debug messages** is off. The Engine records Debug
messages in every build by default.

**Fix:** open **File > Preferences... > General**, under **Logs** tick
**Record Debug messages**. It applies at once and to later starts ("Shown in
the Console under Debug and written to engine_log.txt"). Reproduce the
problem, then collect the log.

## Where the run records are

The Engine keeps a record of the last training run (stage, epoch, batch,
loss, device, how it ended) and the Studio Debugger's runs in a
`debug_runs` folder:

| System | Folder |
| --- | --- |
| Windows | `%LOCALAPPDATA%\CyxWiz\Engine\debug_runs` |
| macOS | `~/Library/Application Support/CyxWiz/Engine/debug_runs` |
| Linux | `$XDG_STATE_HOME/cyxwiz/engine/debug_runs`, or `~/.local/state/cyxwiz/engine/debug_runs` |

`current_run.json` is the last training run. After a crash it still says
how far the run got.

## Where Windows keeps crash reports

Windows Error Reporting writes a short report for each crash under
`%LOCALAPPDATA%\Microsoft\Windows\WER\ReportArchive` (or `ReportQueue`, or
the same folders under `%ProgramData%`), in a folder whose name contains the
program name. The Studio Debugger finds the newest CyxWiz report there and
matches it to the last run.

A full memory dump is written only when Windows Error Reporting's local
dumps are turned on for the program (the `LocalDumps` setting); Windows then
writes it to `%LOCALAPPDATA%\CrashDumps` by default. The Engine does not turn
this on. A dump holds the Engine's memory, including your data: share it
only privately.

## What to attach to a bug report

1. **Help > About CyxWiz...**, then **Copy details**: version, build,
   compute device, Python and licence.
2. Your operating system, and the device and route from
   **File > Preferences... > Devices**.
3. What you did, what you expected, and what you saw, with the exact
   message.
4. `engine_log.txt`, or `engine_log.1.txt` after a crash, with **Record
   Debug messages** on.
5. For a compile or training problem: **Copy report** in the compile report
   window.
6. The graph (`.cyxgraph`) and a small sample of the data, when you may
   share them.

Do not attach passwords, API keys, personal data or data you may not share.
Select **Help > Report Issue...** to open the issue page. Report a security
problem privately instead: see
[Report a vulnerability](../security/report-a-vulnerability.md).

## Related

- [Console and logs](../main-window/console.md)
- [The Studio Debugger](../user-guide/training/studio-debugger.md)
- [Preferences](../main-window/preferences.md)
- [Known issues](../known-issues.md)
