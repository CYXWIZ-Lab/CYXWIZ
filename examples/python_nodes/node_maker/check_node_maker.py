# SPDX-License-Identifier: Apache-2.0
# Part of the CyxWiz plugin SDK examples (see cmake/sdk/NOTICE).
"""Gradient check of the three Make a Python node starters in nodes_src/.

    python check_node_maker.py <folder holding pycyxwiz> [<folder of its DLLs>]

Uses the same pycyxwiz.nodes._engine_check the Engine's Check button runs.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
folder = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
dll_folder = sys.argv[2] if len(sys.argv) > 2 else folder
for d in (folder, dll_folder, r"C:\Program Files\ArrayFire\v3\lib"):
    if os.path.isdir(d) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(d)
os.environ["PATH"] = os.pathsep.join([folder, dll_folder]) + os.pathsep + os.environ.get("PATH", "")
sys.path.insert(0, folder)

import pycyxwiz  # noqa: E402,F401
from pycyxwiz import nodes  # noqa: E402

failures = 0
for name in ("soft_clip.py", "swish.py", "log_cosh.py"):
    for result in json.loads(nodes._engine_check(os.path.join(HERE, "nodes_src", name))):
        ok = result["status"] == "passed"
        failures += 0 if ok else 1
        print("%-4s %s: %s, largest relative error %.2g" % ("ok" if ok else "FAIL", result["name"], result["status"],
                                                             result["max_relative_error"]))
sys.exit(1 if failures else 0)
