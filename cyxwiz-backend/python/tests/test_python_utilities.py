# SPDX-License-Identifier: Apache-2.0
# Part of the CyxWiz plugin SDK examples (see cmake/sdk/NOTICE).
# TOFIX129 D12.
"""pycyxwiz.utilities against Python's own libraries.

Hashes against hashlib, JSON against json, regex matches against re (on
patterns ECMAScript and Python read the same way), unit conversions and the
seeded random values against cyxwiz-engine/tests/computation_truth/fixtures/
utilities_truth.json (exact unit definitions; numpy statistics of the
generators reproduced in Python). Run with the Python pycyxwiz was built for:

    python test_python_utilities.py <folder holding pycyxwiz> [<folder of its DLLs>]

Exit code 0 when every check passes.
"""
import hashlib
import json
import math
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pycyxwiz_test_env import Checks, import_pycyxwiz  # noqa: E402

cyx = import_pycyxwiz()
check = Checks()
u = cyx.utilities

here = os.path.dirname(os.path.abspath(__file__))
fixture = os.path.join(here, "..", "..", "..", "cyxwiz-engine", "tests", "computation_truth", "fixtures", "utilities_truth.json")
with open(fixture, encoding="utf-8") as f:
    truth = json.load(f)


def near(a, b, tolerance):
    return abs(a - b) <= tolerance * max(1.0, abs(b))


def raises(error, call):
    try:
        call()
    except error:
        return True
    return False


# ---- Calculator
check(near(u.evaluate("2 * pi + sqrt(16)"), 2 * math.pi + 4, 1e-15), "evaluate")
check(near(u.evaluate("sin(30)", degrees=True), 0.5, 1e-15), "evaluate in degrees")
check(u.evaluate("rate * 1e4", {"rate": 0.025}) == 250.0, "evaluate with a variable")
check(raises(ValueError, lambda: u.evaluate("2 +")), "a broken expression raises")

# ---- Hash: hashlib on texts, bytes and a file
for text in ["", "abc", "Hello, World!", "héllo ✓", "a" * 5000]:
    for algorithm in ["md5", "sha1", "sha256", "sha512"]:
        check(u.hash(text, algorithm) == hashlib.new(algorithm, text.encode("utf-8")).hexdigest(), f"{algorithm} of {ascii(text[:12])}")
blob = bytes(range(256)) * 9
check(u.hash(blob, "SHA256") == hashlib.sha256(blob).hexdigest(), "bytes, algorithm in any case")
check(raises(ValueError, lambda: u.hash("abc", "sha3")), "an unknown algorithm raises")
with tempfile.TemporaryDirectory() as folder:
    path = os.path.join(folder, "data.bin")
    data = bytes((i * 31 + 7) % 256 for i in range(2 * 1024 * 1024 + 99))
    with open(path, "wb") as out:
        out.write(data)
    seen = []
    digest = u.hash_file(path, "sha512", lambda done, total: seen.append((done, total)) or True)
    check(digest == hashlib.sha512(data).hexdigest(), "hash_file against hashlib")
    check(len(seen) == 3 and seen[-1] == (len(data), len(data)), f"progress per 1 MB block: {seen}")
    check(raises(ValueError, lambda: u.hash_file(path, "md5", lambda done, total: False)), "a stop raises")
    check(raises(ValueError, lambda: u.hash_file(os.path.join(folder, "missing.bin"))), "a missing file raises")

# ---- Units: the fixture's exact conversions, refusals and names
cases = truth["units"]["cases"]
for c in cases[::7]:
    check(near(u.convert(c["value"], c["from"], c["to"]), c["expected"], 1e-12), f'{c["value"]} {c["from"]} -> {c["to"]}')
for r in truth["units"]["refusals"]:
    check(raises(ValueError, lambda r=r: u.convert(r["value"], r["from"], r["to"])), "refused: " + r["reason"])
for name, category, unit_id in truth["units"]["aliases"]:
    found = u.find_unit(name, category)
    check(found is not None and found["id"] == unit_id, f"find_unit({name!r})")
check(u.find_unit("parsec") is None, "no such unit")
check(set(u.unit_categories()) == {"length", "mass", "temperature", "time", "data", "area", "volume", "speed"}, "categories")
check(all(x["category"] == "data" for x in u.units("data")) and len(u.units()) > 70, "the unit table")
check(u.conversion_formula("celsius", "fahrenheit") == "°F = °C x 1.8 + 32", "the formula reads 1.8 and 32")
check(u.convert(1, "kilobyte", "byte") == 1000 and u.convert(1, "kibibyte", "byte") == 1024, "kB and KiB")

# ---- JSON: Python's json on the same texts
for c in truth["json"]["valid"]:
    r = u.json_check(c["text"], 4)
    check(r["valid"] and json.loads(r["minified"]) == json.loads(c["text"]), "valid json round trip")
    check(r["keys"] == c["keys"] and r["depth"] == c["depth"] and r["objects"] == c["objects"], "json counts")
    check(r["formatted"] == json.dumps(json.loads(c["text"]), indent=4, ensure_ascii=False)
          or json.loads(r["formatted"]) == json.loads(c["text"]), "formatted keeps the value")
for c in truth["json"]["invalid"]:
    r = u.json_check(c["text"])
    check(not r["valid"] and r["line"] == c["line"] and r["column"] == c["column"],
          f'error place of {ascii(c["text"])}: {r.get("line")}, {r.get("column")}')
doc = '{"b": 1, "a": {"list": [10, {"name": "x"}]}, "s": "text"}'
check(list(json.loads(u.json_check(doc)["minified"]).keys()) == ["b", "a", "s"], "keys keep the document's order")
check(u.json_get(doc, "a.list[1].name") == "x" and u.json_get(doc, "$.a.list[0]") == 10, "json_get values")
check(u.json_get(doc, "a") == {"list": [10, {"name": "x"}]} and u.json_get(doc, "s") == "text", "json_get objects and strings")
check(raises(KeyError, lambda: u.json_get(doc, "a.list[5]")) and raises(KeyError, lambda: u.json_get(doc, "nope")),
      "a missing path raises KeyError")

# ---- Regex: Python's re on the fixture's text
text = truth["regex"]["text"]
for c in truth["regex"]["cases"]:
    flags = c["flags"]
    got = u.regex_find(c["pattern"], text, ignore_case="i" in flags, multiline="m" in flags)
    want = [(m.group(0), m.start(), m.end(), list(m.groups()))
            for m in re.finditer(c["pattern"], text, (re.I if "i" in flags else 0) | (re.M if "m" in flags else 0))]
    check([(g["text"], g["start"], g["end"], g["groups"]) for g in got] == want, f'regex {c["pattern"]} /{flags}')
unicode_text = "café one\nnaïve two"
got = u.regex_find("one|two", unicode_text)
check([(g["start"], g["end"], g["line"], g["column"]) for g in got] == [(5, 8, 1, 6), (15, 18, 2, 7)],
      f"positions in characters, as re: {ascii(got)}")
check([m.start() for m in re.finditer("one|two", unicode_text)] == [5, 15], "re agrees")
check(u.regex_replace(r"(\d{3})[-.](\d{4})", text, "$2-$1") == re.sub(r"(\d{3})[-.](\d{4})", r"\2-\1", text), "replace")
check(raises(ValueError, lambda: u.regex_find("(", text)), "a bad pattern raises")

# ---- Random: the fixture's values and numpy statistics
for c in truth["random"]:
    a, b = c["a"], c["b"]
    r = u.random(c["kind"], c["count"], a, b, c["seed"])
    tag = f'{c["kind"]} seed {c["seed"]}'
    check(all(near(x, y, 1e-13) for x, y in zip(r["values"][:10], c["first"])), tag + ": first values")
    check(near(r["mean"], c["mean"], 1e-12) and near(r["sd_sample"], c["std_sample"], 1e-10)
          and near(r["sd_population"], c["std_population"], 1e-10), tag + ": mean and SDs")
    check(near(r["median"], c["median"], 1e-12) and near(r["q1"], c["q1"], 1e-12) and near(r["q3"], c["q3"], 1e-12),
          tag + ": quartiles")
    check(r["counts"] == c["counts"] and r["seed"] == c["seed"], tag + ": histogram counts and seed")
check(raises(ValueError, lambda: u.random("gamma", 10)), "an unknown distribution raises")
check(raises(ValueError, lambda: u.random("normal", 10, 0.0, -1.0, 1)), "a negative SD raises")
drawn = u.random("uniform", 5)
check(drawn["seed"] >= 0 and len(drawn["values"]) == 5, "seed -1 draws a seed and returns it")
ids = u.uuid4(50)
check(len(set(ids)) == 50 and all(i[14] == "4" and i[19] in "89ab" for i in ids), "uuid4: version 4, all different")

sys.exit(check.finish("pycyxwiz.utilities matches hashlib, json, re and the utilities fixture"))
