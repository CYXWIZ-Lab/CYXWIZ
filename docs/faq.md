---
title: Frequently asked questions
description: Short answers to common questions about installing, GPUs, data, Python, plugins, remote training, the licence, privacy and versions.
type: reference
area: general
audience: user
status: stable
since: 1.0.7
owner: general
last_reviewed: 2026-11-25
related:
  - troubleshooting/index.md
  - known-issues.md
  - glossary.md
  - install/index.md
---

Each answer is short and links the page with the detail. For an error
message, see [Troubleshooting](troubleshooting/index.md).

## The Engine

### What is the CyxWiz Engine?

A desktop application for machine learning. You build a model as a graph of
nodes on a canvas, prepare data in Data Studio, train on your CPU or GPU,
then test and export the model. Start with
[Start the Engine for the first time](install/first-launch.md).

### Do I need to write code?

No. Graphs, Data Studio, training and evaluation work without code. Python
is there when you want it: scripts, the Python console, Python nodes, and
code export of a graph to PyTorch, TensorFlow, Keras or PyCyxWiz. See
[The Studio bar](main-window/toolbar.md).

### Is there a quick example to start from?

Yes. The start page lists **Task starter graphs** (image, text, audio and
time-series), each with a small sample dataset, so it trains straight away.
More graphs are in the `examples/` folder of the install.

### Can I work without a project?

Yes: **Continue without project** on the start page. **File > Save** is not
available then, so create a project before you build something to keep. See
[The start page](main-window/start-page.md).

## Install and systems

### Which systems does it run on?

Windows x64, Linux x86_64, and macOS on Intel and Apple Silicon. Releases
are built on Windows Server 2022, Ubuntu 22.04 and macOS 15; older systems
are untested. See [Install CyxWiz](install/index.md).

### Do I need a GPU?

No. The Engine always computes on the CPU, through ArrayFire's CPU runtime.
A GPU makes training faster when a GPU pack is installed and verified.

### Which GPUs are supported?

On Windows x64: NVIDIA (CUDA or OpenCL) and Intel (OpenCL or oneAPI)
through GPU packs. AMD GPUs through OpenCL are untested. Linux and macOS
installs are CPU only for now. See
[Add a GPU pack and verify it](install/gpu-packs.md).

### I installed a GPU pack. Why does training still use the CPU?

Training uses only routes that passed verification on your machine. Open
**Preferences > Devices**, select **Verify all**, then **Use this device** on
the GPU route. See [GPU and driver problems](troubleshooting/gpu-and-drivers.md).

### Does the installer install GPU drivers?

No. A pack adds the GPU runtime beside the Engine, in the install folder;
you install the driver from your GPU vendor.

### How do I update or uninstall?

With the **CyxWiz Installer**. See
[Update or uninstall](install/update-and-uninstall.md).

## Data

### Which file formats can I load?

Tables such as CSV, TSV and Parquet, image and audio folders, text, HDF5,
and formats that plugins add. The Data Convert node converts between CSV,
TSV, JSON Lines, Parquet, Feather, Arrow IPC, Excel, ARFF, NumPy and HDF5.
See [Data Input formats](main-window/data-input-formats.md).

### How large can a dataset be?

Larger than memory. A table that does not fit the **data budget** stays on
disk and training reads it in pages. See
[Data bigger than memory](concepts/data-bigger-than-memory.md).

### Why does Train say my data is not loaded after I reopened a project?

A project keeps each Data Input's settings, not its loaded data. Train
loads the data itself; to load it earlier, open the Data Input and click
**Apply**. See [Loaded data](concepts/loaded-data.md).

### Can a preparation step leak test data into training?

A step before the Data Split that learns from the data (a scaler, an
encoder) learns from every row, and the compile report warns about it. See
[Steps learn from the Train rows](concepts/steps-learn-from-train-rows.md).

### Where can my datasets live besides my computer?

In CyxCloud: a Data Input can name a file in a CyxCloud dataset. See
[Datasets that live in CyxCloud](server-and-cloud/cyxcloud/datasets.md).

## Python

### Which Python do I need?

Python 3.12 with its `venv` module, for scripts, the Python console and
project environments. The Engine embeds Python 3.12, so other versions are
listed as not usable. See [Python for scripting](install/first-launch.md#python-for-scripting).

### Can I use NumPy, pandas or other packages?

Yes, in the project's environment: the Engine creates a virtual
environment (`python/` in the project folder) from the Python you chose the
first time a project is created or opened, and you add packages to it. See
[Notebooks in the Script Editor](user-guide/scripting/notebooks.md) for how projects and
their environments switch.

## Training and results

### Are my results the same as PyTorch?

Close, not bit for bit. The Engine's layers and losses are checked against
PyTorch as the reference, to float32 precision. See
[Numerical accuracy](concepts/numerical-accuracy.md).

### Does a seeded run repeat?

Yes on the same device and build. Across systems and devices the random
draws are the same, but GPU sums are added in a different order, so losses
agree closely, not exactly. See
[Seeds and reproducibility](concepts/seeds-and-reproducibility.md).

### Training stopped. Can I resume it?

Yes, from a checkpoint. See
[Checkpoints, early stopping and resume](user-guide/training/checkpoints-and-early-stopping.md).

### Which formats can I export a model to?

A CyxWiz package (`.cyxmodel`), Safetensors, and ONNX for layer stacks in
builds with ONNX support. See [Model formats](concepts/model-formats.md).

### Can I start from PyTorch weights?

Yes, from PyTorch and Safetensors weight files. See
[Start from pretrained weights](user-guide/training/start-from-pretrained-weights.md).

## Plugins and extensions

### Can I add my own nodes?

Yes, as a C++ plugin built with the CyxWiz Plugin SDK, or as a Python node
bundle. See [Your own nodes](usage/extension_nodes.md) and the
[plugin quickstart](sdk/get-started/first-cpp-plugin.md).

### Is it safe to open a graph or plugin someone sent me?

Treat them as programs. A plugin runs native code with your rights, and a
Python node runs Python code. See
[Untrusted files and plugins](security/untrusted-files-and-plugins.md).

### May I sell a closed-source plugin?

Yes. The SDK is under the Apache License 2.0, and a plugin that uses only
the SDK's interfaces is not a work based on the Engine. See the
[Licensing FAQ](legal/licensing-faq.md).

## Remote training

### Can I train on another computer?

Yes. A computer running the CyxWiz Server Node offers its device; from the
Engine you reserve it, connect and train your graph there. See
[Server Connection](server-and-cloud/server-address.md). Read
[Security](security/index.md) before you run a node on a
network you do not control.

### Can one training run use several machines at once?

No. A training job runs on one node.

## Licence, privacy and versions

### Is the Engine free?

Yes. The Engine and the Server Node are free software under the GNU Affero
General Public License, version 3 or later. A commercial licence is offered
for closed products and hosted services that do not want the AGPL's
obligations. See the [Licensing FAQ](legal/licensing-faq.md).

### Does the Engine send data about me?

It sends no telemetry and checks for no updates by itself. It contacts a
server only for a feature you use, such as signing in, CyxCloud or remote
training. See [Privacy](about/privacy.md) for exactly what is sent.

### Which version gets fixes?

CyxWiz is in alpha: only the latest release gets fixes. See
[Support and versioning](about/support-and-versioning.md).

### Where are my settings and logs?

Settings: `engine_config.json` in `%APPDATA%\CyxWiz` on Windows,
`~/.cyxwiz` on Linux and macOS. Logs: `engine_log.txt` beside the Engine
program (**Help > Open Logs Folder**). See
[Crash dumps and logs](troubleshooting/crash-dumps-and-logs.md).

### How do I report a bug or a vulnerability?

A bug: **Help > Report Issue...**, with what is listed in
[What to attach to a bug report](troubleshooting/crash-dumps-and-logs.md#what-to-attach-to-a-bug-report).
A vulnerability: privately, as in
[Report a vulnerability](security/report-a-vulnerability.md).

## Related

- [Troubleshooting](troubleshooting/index.md)
- [Known issues](known-issues.md)
- [Glossary](glossary.md)
