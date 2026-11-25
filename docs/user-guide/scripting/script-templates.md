---
title: Script templates
description: The starter scripts in the Engine's scripts/templates folder, what each one is for, and how to start a script from one.
type: how-to
area: scripting
audience: user
status: experimental
since: 1.0.7
owner: scripting
last_reviewed: 2026-11-25
related:
  - script-editor.md
  - cyx-documents.md
  - ../../sdk/guides/python-modules.md
---

Use this when you want a starting point for a script instead of an empty
file. The templates are `.cyx` files with sections and `# TODO:` markers
where you fill in your own paths and values.

## Before you start

- The templates are in the `scripts/templates` folder of the Engine's
  source repository, and a build from source copies them next to the Engine
  (`scripts/templates` beside the executable). The release packages do not
  include them.
- The Engine has no template picker: you start from a copy of the file.

## The templates

| File | Starting point for |
| --- | --- |
| `data_loading.cyx` | Loading CSV, tab- or space-separated text and HDF5 files and printing what they hold |
| `data_processing.cyx` | Cleaning and transforming data, simple feature engineering |
| `plotting.cyx` | Plots with matplotlib |
| `custom_function.cyx` | A small library of documented helper functions |
| `model_training.cyx` | Building and training a model with `pycyxwiz` (see the limit below) |
| `kaggle_datasets.cyx` | Downloading a dataset with the `kaggle` package (needs a Kaggle account and its API key) |

## Start a script from a template

1. Copy the template into your project's `scripts` folder and give the copy
   your own name, for example `load_sales.cyx`.
2. In the Engine, double-click the copy in the Asset Browser. It opens in the
   Script Editor.
3. Press **Ctrl+F** and search for `TODO:`. Each one says what to change:
   file names, columns, parameters.
4. Press **F5** to run the script. Its output shows in the Console's
   **Python REPL** session.

The templates import optional packages (pandas, matplotlib) inside
`try` blocks and print a warning when one is missing. To add one, see
[Install packages](python-environment.md#install-packages).

## Limits

- `model_training.cyx` was written for an older `pycyxwiz`: it calls
  `cyx.Tensor.randn`, which the current module does not have, and its model
  and optimizer lines are comments. For training from a script, start from
  [Write a module in Python](../../sdk/guides/python-modules.md) and the
  [Python API reference](../../sdk/python-api/index.md) instead.
- The committed, runnable scripts in the repository's `examples/python`
  folder (shipped in the release packages under `examples/python`) are
  better starting points for most tasks.

## Related

- [Write and run scripts](script-editor.md)
- [The .cyx document format](cyx-documents.md)
- [Startup scripts](startup-scripts.md)
