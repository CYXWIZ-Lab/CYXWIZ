---
title: CyxWiz Engine documentation
description: Start here to learn the CyxWiz Engine with a tutorial, install it, learn its window, build a model in the graph editor and look up any part of it.
type: landing
area: general
audience: user
status: stable
since: 1.0.7
owner: docs
last_reviewed: 2026-11-25
related:
  - tutorials/index.md
  - install/index.md
  - main-window/index.md
  - user-guide/graph-editor/index.md
  - reference/shortcuts.md
redirect_from: []
---

## Get started with a tutorial

Train your first model in about 15 minutes: open a starter graph, look at
each node, press **Train** and read the results.

[Training a model in CyxWiz Engine, 6x speed (video)](https://engine.cyxwiz.ai/media/v/mnist-session.mp4)

1. [Tour the main window](tutorials/tour-the-main-window.md)
2. [Train your first model from a starter graph](tutorials/first-model-from-a-starter.md)
3. [Bring in and prepare your own data](tutorials/prepare-your-own-data.md)

The [learning path](tutorials/index.md) has six lessons, then lessons on
text, audio, time series and Python.

## Install and first launch

The CyxWiz Engine is a desktop application for building machine-learning
models as graphs: you place nodes on a canvas, connect them, set their
properties, and train, debug, test and export the result on your own
computer.

| | |
| --- | --- |
| [Install the Engine](install/index.md) | Windows, Linux and macOS packages, GPU packs, update and uninstall |
| [First launch](install/first-launch.md) | What you see the first time, Python for scripting, your first graph |
| [The start page](main-window/start-page.md) | Starter graphs, recent projects, new projects |

## Using the Engine

| | |
| --- | --- |
| [The main window](main-window/index.md) | The parts of the window, panels and the sidebar |
| [Menus](main-window/menus.md) | Every menu, File to Help |
| [The Studio bar](main-window/toolbar.md) | The bar above the canvas: graph, mode, Compile, Local Debug, Train |
| [The status bar](main-window/status-bar.md) | Project, Python setup, hints, tasks |
| [Layouts and docking](main-window/layouts-and-docking.md) | Arrange panels, windows of their own, reset the layout |
| [Editing values](main-window/editing-values.md) | Typed number fields, steps, dropdowns, file and column pickers |
| [Preferences](main-window/preferences.md) | The Preferences dialog |

## Build a model

| | |
| --- | --- |
| [The graph editor](user-guide/graph-editor/index.md) | From an empty canvas to a trained model |
| [The canvas](user-guide/graph-editor/canvas.md) | Pan, zoom, select, frames, groups, undo |
| [Find and add nodes](user-guide/graph-editor/find-and-add-nodes.md) | The Nodes panel, search, the right-click menu |
| [Connect nodes](user-guide/graph-editor/connect-nodes.md) | Pins, which connect, refused links |
| [Set node properties](user-guide/graph-editor/node-properties.md) | The Properties panel and node dialogs |
| [Compile and run](user-guide/graph-editor/compile-and-run.md) | Compile, the compile report, run modes |
| [The Studio Debugger](user-guide/training/studio-debugger.md) | Local Debug and the layer views |

## Script, extend and run elsewhere

| | |
| --- | --- |
| [Scripting](user-guide/scripting/index.md) | The Script Editor, Python console, Variable Explorer, notebooks, the Python environment |
| [Extend CyxWiz](sdk/index.md) | Your own nodes in Python or C++, the plugin SDK and the pycyxwiz reference |
| [Server node and CyxCloud](server-and-cloud/index.md) | Run a server node, CyxCloud storage, server addresses, ports and security |
| [Remote training](distributed-training/index.md) | Remote jobs today, how data reaches the node, and their limits |

## Look things up

| | |
| --- | --- |
| [Keyboard shortcuts](reference/shortcuts.md) | Every shortcut, by where it works |
| [Error messages](reference/error-messages.md) | Every CW code: what it means and what to do |
| [Glossary](glossary.md) | The terms these docs use |
| [User guide (older pages)](usage/README.md) | Data Studio, training, evaluation, scripting, plugins and more, while they move into this structure |
| [Release notes](RELEASE_NOTES.md) | What changed |

## How it works

| | |
| --- | --- |
| [How CyxWiz works](concepts/index.md) | The ecosystem, how graphs run, the data pipeline, compute back ends, accuracy, model files |
| [The ecosystem](concepts/ecosystem.md) | The Engine, the server node, CyxCloud and the central server, and how they connect |

## Get help

| | |
| --- | --- |
| [Troubleshooting](troubleshooting/index.md) | By symptom: the message you see, the cause, the fix |
| [Known issues](known-issues.md) | Problems we know about, the devices they affect, what to do |
| [FAQ](faq.md) | Short answers to common questions |
| [Support and versioning](about/support-and-versioning.md) | Where to ask, what to include, which releases get fixes |

## Security, licence and privacy

| | |
| --- | --- |
| [Security](security/index.md) | What opened files, plugins, Python and remote jobs can do; reporting a vulnerability |
| [Licensing FAQ](legal/licensing-faq.md) | The AGPL, the commercial licence and the plugin SDK |
| [Privacy](about/privacy.md) | What the Engine sends, and when |

## Contribute

| | |
| --- | --- |
| [Engine internals](internals/index.md) | Architecture, code layout, build, tests, adding a node or a panel |

## About these docs

The docs describe what the current build does, including its limits. They
are written in the Engine's source repository and checked against its
code. To report a mistake, use **Help > Report Issue...** in the Engine.
