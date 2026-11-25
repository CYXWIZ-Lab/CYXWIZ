#!/usr/bin/env python3
"""Proves that tests leave the user's own CyxWiz folders alone.

Test executables keep their settings, logs and caches in a scratch folder of
their own (cyxwiz-engine/tests/support/scratch_user_folders.cpp). CI records
the user's CyxWiz folders before the test steps and checks them after:

    python scripts/checks/check_user_folders_untouched.py record <state.json>
    ... the test steps ...
    python scripts/checks/check_user_folders_untouched.py check <state.json>

`check` fails, listing each path, when a file or folder was added, changed or
removed. The folders are those of cyxwiz-engine/src/core/user_folders.h.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys


def user_folders() -> list[pathlib.Path]:
    home = pathlib.Path.home()
    if sys.platform == "win32":
        folders = []
        for name in ("LOCALAPPDATA", "APPDATA"):
            value = os.environ.get(name)
            if value:
                folders.append(pathlib.Path(value) / "CyxWiz")
        return folders
    if sys.platform == "darwin":
        library = home / "Library"
        return [home / ".cyxwiz", library / "Application Support" / "CyxWiz",
                library / "Logs" / "CyxWiz", library / "Caches" / "CyxWiz"]
    state = pathlib.Path(os.environ.get("XDG_STATE_HOME") or home / ".local" / "state")
    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME") or home / ".cache")
    return [home / ".cyxwiz", state / "cyxwiz", cache / "cyxwiz"]


def listing() -> dict[str, str]:
    entries: dict[str, str] = {}
    for folder in user_folders():
        if not folder.exists():
            continue
        entries[str(folder)] = "folder"
        for path in sorted(folder.rglob("*")):
            try:
                info = path.lstat()
            except OSError:
                continue
            kind = "folder" if path.is_dir() else f"{info.st_size} bytes"
            entries[str(path)] = f"{kind}, modified {info.st_mtime_ns}"
    return entries


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("record", "check"):
        print(__doc__)
        return 2
    state = pathlib.Path(argv[2])
    if argv[1] == "record":
        state.parent.mkdir(parents=True, exist_ok=True)
        entries = listing()
        state.write_text(json.dumps(entries, indent=1, sort_keys=True), encoding="utf-8")
        print(f"Recorded {len(entries)} entries in the user's CyxWiz folders:")
        for folder in user_folders():
            print(f"  {folder}")
        return 0
    before = json.loads(state.read_text(encoding="utf-8"))
    after = listing()
    problems = [f"added: {path}" for path in sorted(after.keys() - before.keys())]
    problems += [f"removed: {path}" for path in sorted(before.keys() - after.keys())]
    problems += [f"changed: {path}" for path in sorted(after.keys() & before.keys())
                 if after[path] != before[path]]
    if problems:
        print("The tests wrote into the user's CyxWiz folders; they must use their scratch folder:")
        for line in problems:
            print(f"  {line}")
        return 1
    print(f"The user's CyxWiz folders are unchanged ({len(after)} entries).")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
