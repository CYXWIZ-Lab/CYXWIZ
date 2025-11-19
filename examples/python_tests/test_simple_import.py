import sys
import os

dll_dir1 = "D:/Dev/CyxWiz_Claude/build/windows-release/bin/Release"
dll_dir2 = "D:/Dev/CyxWiz_Claude/build/windows-release/lib/Release"

os.environ['PATH'] = dll_dir1 + os.pathsep + dll_dir2 + os.pathsep + os.environ.get('PATH', '')
sys.path.insert(0, dll_dir2)

try:
    import pycyxwiz as cx
    print("Import successful!")
    cx.initialize()
    print("Version:", cx.get_version())
    cx.shutdown()
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
