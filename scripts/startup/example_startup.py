# An example startup script for the Engine's Python sessions.
#
# Add it in Preferences > Python > When Python starts > Startup scripts >
# Add a script... It runs in each Python session before the session's first
# command or script, and again after Restart, in the session's namespace:
# what it defines is there in the Python console and in scripts. What it
# prints goes to the Console log ("Startup script printed: ...").
import json
import math
from pathlib import Path


def read_json(path):
    """Read a JSON file into Python values."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def describe(values):
    """Count, mean and standard deviation of a list of numbers."""
    n = len(values)
    if n == 0:
        return {"count": 0, "mean": None, "sd": None}
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / n)
    return {"count": n, "mean": mean, "sd": sd}


print("Startup helpers ready: read_json(path), describe(values)")
