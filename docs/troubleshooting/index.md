---
title: Troubleshooting
description: Find the fix for a message or a problem by what you see, from the installer to a crashed training run.
type: landing
area: general
audience: user
status: stable
since: 1.0.7
owner: general
last_reviewed: 2026-11-25
related:
  - ../known-issues.md
  - ../reference/error-messages.md
  - ../faq.md
  - crash-dumps-and-logs.md
---

Each page here is organised by symptom. A heading is the exact text the
Engine shows, so you can search for the message you see. Each entry gives
the cause and the fix.

## Find your symptom

| You see a problem with | Page |
| --- | --- |
| The installer, the first start, Python for scripting | [Install and first launch](install-and-first-launch.md) |
| A GPU route, verification, drivers, training on the CPU | [GPU and drivers](gpu-and-drivers.md) |
| Loading a file in a Data Input, memory, the data budget | [Data loading](data-loading.md) |
| Compile findings, Train greyed out, "Cannot start training" | [Compile and training](compile-and-training.md) |
| A plugin that does not load, a Python node | [Plugins and Python](plugins-and-python.md) |
| A crash, or you need logs for a bug report | [Crash dumps and logs](crash-dumps-and-logs.md) |

A message that starts with a code such as `[CW-C-0201]` is a diagnostic
code. Every code is listed with its meaning in
[Error messages](../reference/error-messages.md).

## Before you search

1. Read the whole message. Most Engine messages end with what to do, for
   example "open the node and click Apply".
2. Open the **Console** (bottom dock) and look at the **Warn** and **Error**
   lines around the time it happened. Click a row to see the whole message.
3. Check [Known issues](../known-issues.md): some problems are in the GPU
   runtimes the Engine builds on, with a workaround.

## When nothing here helps

Open an issue with the details listed in
[What to attach to a bug report](crash-dumps-and-logs.md#what-to-attach-to-a-bug-report).
Report security problems privately, as described in
[Report a vulnerability](../security/report-a-vulnerability.md).

## Related

- [Known issues](../known-issues.md)
- [Error messages](../reference/error-messages.md)
- [Frequently asked questions](../faq.md)
- [Troubleshooting extension nodes](../sdk/guides/troubleshooting.md)
