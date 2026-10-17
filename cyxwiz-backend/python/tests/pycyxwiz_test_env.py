"""Imports the pycyxwiz under test for the scripts in this folder.

    python <test>.py <folder holding pycyxwiz> [<folder of its DLLs>]
"""
import os
import sys


def import_pycyxwiz():
    folder = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    dll_folder = sys.argv[2] if len(sys.argv) > 2 else folder
    dirs = [folder, dll_folder, r"C:\Program Files\ArrayFire\v3\lib"]
    if os.environ.get("CUDA_PATH"):
        dirs.append(os.path.join(os.environ["CUDA_PATH"], "bin"))
    for d in dirs:
        if os.path.isdir(d) and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(d)
    # ArrayFire loads its backend libraries through PATH.
    os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")
    sys.path.insert(0, folder)
    import pycyxwiz
    return pycyxwiz


class Checks:
    def __init__(self):
        self.failures = []

    def __call__(self, condition, message):
        print(("ok   " if condition else "FAIL ") + message)
        if not condition:
            self.failures.append(message)

    def finish(self, name):
        if self.failures:
            print("%d check(s) failed" % len(self.failures))
            sys.exit(1)
        print(name + ": all checks passed")
