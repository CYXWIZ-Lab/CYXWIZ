"""Truth for cyxwiz::Utilities (TOFIX129 D12, Tools > Utilities), for
test_computation_truth_utilities.

  hash     hashlib md5 / sha1 / sha256 / sha512 of texts (empty, ASCII,
           UTF-8) and of a 3 MiB byte pattern (byte i = (i * 31 + 7) % 256,
           written by the test, so the file spans several 1 MiB blocks)
  units    every unit of the table converted to its category's base and back,
           from the exact definitions (fractions.Fraction): international
           yard and pound, nautical mile 1852 m, US gallon 231 in^3, imperial
           gallon 4.54609 L, 0 degC = 273.15 K, degF = 5/9 K; and refusals
  json     Python's json on valid texts (counts of objects, arrays, strings,
           numbers, booleans, nulls, top-level keys, depth) and on invalid
           texts where Python and nlohmann stop at the same character (line,
           column in characters)
  regex    Python's re (ECMAScript-compatible patterns): spans in characters
           of ASCII text, with and without IGNORECASE / MULTILINE, and
           re.sub replacements
  random   Utilities::Generate* reproduced: std::mt19937_64 (written out
           below, the standard's algorithm) and cyxwiz::portable's draws, the
           first values, then numpy's mean, std (ddof 0 and 1), percentiles
           25 / 50 / 75 (linear) and histogram counts on the backend's edges

Run from this folder: python generate_utilities_fixtures.py
"""
import hashlib
import json
import math
import pathlib
import re
from fractions import Fraction as F

import numpy as np

OUT = pathlib.Path(__file__).resolve().parent.parent / "fixtures" / "utilities_truth.json"

# ---------------------------------------------------------------- hash
TEXTS = ["", "abc", "Hello, World!", "héllo ✓ über", "a" * 1000,
         "The quick brown fox jumps over the lazy dog"]
FILE_BYTES = 3 * 1024 * 1024 + 123


def file_pattern(n):
    return bytes(((i * 31 + 7) % 256) for i in range(n))


def digests(data):
    return {a: hashlib.new(a, data).hexdigest() for a in ("md5", "sha1", "sha256", "sha512")}


hash_cases = [{"text": t, **digests(t.encode("utf-8"))} for t in TEXTS]
hash_file = {"bytes": FILE_BYTES, **digests(file_pattern(FILE_BYTES))}

# ---------------------------------------------------------------- units
INCH, FOOT, YARD, MILE, POUND = F("0.0254"), F("0.3048"), F("0.9144"), F("1609.344"), F("0.45359237")
US_GALLON = F(231) * INCH ** 3 * 1000  # litres
IMP_GALLON = F("4.54609")
UNITS = {  # id: (category, scale, offset): base = value * scale + offset
    "meter": ("length", F(1), 0), "kilometer": ("length", F(1000), 0), "centimeter": ("length", F(1, 100), 0),
    "millimeter": ("length", F(1, 1000), 0), "micrometer": ("length", F(1, 10**6), 0),
    "nanometer": ("length", F(1, 10**9), 0), "inch": ("length", INCH, 0), "foot": ("length", FOOT, 0),
    "yard": ("length", YARD, 0), "mile": ("length", MILE, 0), "nautical_mile": ("length", F(1852), 0),
    "kilogram": ("mass", F(1), 0), "gram": ("mass", F(1, 1000), 0), "milligram": ("mass", F(1, 10**6), 0),
    "tonne": ("mass", F(1000), 0), "pound": ("mass", POUND, 0), "ounce": ("mass", POUND / 16, 0),
    "stone": ("mass", 14 * POUND, 0), "short_ton": ("mass", 2000 * POUND, 0), "long_ton": ("mass", 2240 * POUND, 0),
    "nanosecond": ("time", F(1, 10**9), 0), "microsecond": ("time", F(1, 10**6), 0),
    "millisecond": ("time", F(1, 1000), 0), "second": ("time", F(1), 0), "minute": ("time", F(60), 0),
    "hour": ("time", F(3600), 0), "day": ("time", F(86400), 0), "week": ("time", F(604800), 0),
    "year": ("time", F(365 * 86400), 0),
    "bit": ("data", F(1, 8), 0), "byte": ("data", F(1), 0), "kilobit": ("data", F(125), 0),
    "megabit": ("data", F(125000), 0), "gigabit": ("data", F(125 * 10**6), 0), "kilobyte": ("data", F(1000), 0),
    "megabyte": ("data", F(10**6), 0), "gigabyte": ("data", F(10**9), 0), "terabyte": ("data", F(10**12), 0),
    "petabyte": ("data", F(10**15), 0), "kibibyte": ("data", F(2**10), 0), "mebibyte": ("data", F(2**20), 0),
    "gibibyte": ("data", F(2**30), 0), "tebibyte": ("data", F(2**40), 0), "pebibyte": ("data", F(2**50), 0),
    "square_meter": ("area", F(1), 0), "square_centimeter": ("area", F(1, 10**4), 0),
    "square_kilometer": ("area", F(10**6), 0), "hectare": ("area", F(10**4), 0), "square_inch": ("area", INCH**2, 0),
    "square_foot": ("area", FOOT**2, 0), "square_yard": ("area", YARD**2, 0), "acre": ("area", 4840 * YARD**2, 0),
    "square_mile": ("area", MILE**2, 0),
    "liter": ("volume", F(1), 0), "milliliter": ("volume", F(1, 1000), 0), "cubic_meter": ("volume", F(1000), 0),
    "cubic_centimeter": ("volume", F(1, 1000), 0), "cubic_inch": ("volume", INCH**3 * 1000, 0),
    "cubic_foot": ("volume", FOOT**3 * 1000, 0), "gallon": ("volume", US_GALLON, 0),
    "quart": ("volume", US_GALLON / 4, 0), "pint": ("volume", US_GALLON / 8, 0), "cup": ("volume", US_GALLON / 16, 0),
    "fluid_ounce": ("volume", US_GALLON / 128, 0), "imperial_gallon": ("volume", IMP_GALLON, 0),
    "imperial_pint": ("volume", IMP_GALLON / 8, 0),
    "meter_per_second": ("speed", F(1), 0), "kilometer_per_hour": ("speed", F(1000, 3600), 0),
    "mile_per_hour": ("speed", MILE / 3600, 0), "knot": ("speed", F(1852, 3600), 0),
    "foot_per_second": ("speed", FOOT, 0),
    "kelvin": ("temperature", F(1), F(0)), "celsius": ("temperature", F(1), F("273.15")),
    "fahrenheit": ("temperature", F(5, 9), F("459.67") * F(5, 9)), "rankine": ("temperature", F(5, 9), F(0)),
}
VALUES = [F(1), F("2.5"), F(-3), F("1234.5678")]
unit_cases = []
for uid, (cat, scale, offset) in UNITS.items():
    peers = [u for u, (c, _, _) in UNITS.items() if c == cat]
    for to in peers:
        for v in VALUES:
            base = v * scale + offset
            if cat == "temperature" and base < 0:
                continue
            _, tscale, toffset = UNITS[to]
            unit_cases.append({"value": float(v), "from": uid, "to": to, "expected": float((base - toffset) / tscale)})
unit_refusals = [
    {"value": -300, "from": "celsius", "to": "kelvin", "reason": "Below absolute zero"},
    {"value": 1, "from": "meter", "to": "kilogram", "reason": "Cannot convert length to mass"},
    {"value": 1, "from": "parsec", "to": "meter", "reason": "Unknown source unit"},
    {"value": 1, "from": "ton", "to": "kilogram", "reason": "Unknown source unit"},
]
unit_aliases = [  # name, category ("" = any) -> id
    ["km/h", "", "kilometer_per_hour"], ["metres", "", "meter"], ["Square Foot", "", "square_foot"],
    ["MiB", "", "mebibyte"], ["kB", "", "kilobyte"], ["degF", "", "fahrenheit"], ["mm", "", "millimeter"],
    ["lbs", "", "pound"], ["t", "mass", "tonne"], ["nmi", "", "nautical_mile"], ["l", "volume", "liter"],
]

# ---------------------------------------------------------------- json


def json_counts(text):
    root = json.loads(text)
    c = {"objects": 0, "arrays": 0, "strings": 0, "numbers": 0, "bools": 0, "nulls": 0, "depth": 0}

    def walk(v, depth):
        c["depth"] = max(c["depth"], depth)
        if isinstance(v, dict):
            c["objects"] += 1
            for x in v.values():
                walk(x, depth + 1)
        elif isinstance(v, list):
            c["arrays"] += 1
            for x in v:
                walk(x, depth + 1)
        elif isinstance(v, bool):
            c["bools"] += 1
        elif v is None:
            c["nulls"] += 1
        elif isinstance(v, (int, float)):
            c["numbers"] += 1
        else:
            c["strings"] += 1
    walk(root, 0)
    c["keys"] = list(root.keys()) if isinstance(root, dict) else []
    c["minified"] = json.dumps(root, separators=(",", ":"), ensure_ascii=False)
    return c


VALID = [
    '{"name": "CyxWiz", "version": "1.0.0", "features": ["Visual Node Editor", "GPU Acceleration"],'
    ' "config": {"backend": "ArrayFire", "gpu_enabled": true, "max_workers": 10}}',
    '[1, 2.5, -3e2, null, true, false, "x", {"a": [[], {}]}]',
    '{"été": "✓", "deep": {"a": {"b": {"c": [0]}}}}',
    '42',
]
INVALID = [
    '{\n  "name": "CyxWiz",\n  "version": "1.0.0"\n  "gpu": true\n}',
    '[1, 2, 3,]',
    '{"a": 1}}',
    '{"été": 1 "x": 2}',
]


def error_place(text):
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        line = text.count("\n", 0, e.pos) + 1
        col = e.pos - (text.rfind("\n", 0, e.pos) + 1) + 1
        return {"text": text, "line": line, "column": col, "python": e.msg}
    raise SystemExit("expected an error: " + text)


json_valid = [{"text": t, **json_counts(t)} for t in VALID]
json_invalid = [error_place(t) for t in INVALID]

# ---------------------------------------------------------------- regex
TEXT = ("Contact Information:\nJohn Doe - john.doe@example.com - (555) 123-4567\n"
        "Jane Smith - jane_smith123@company.org - 555.987.6543\nSupport Team - support@help-desk.io\n")
REGEX = [
    {"pattern": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "flags": ""},
    {"pattern": r"^\w+", "flags": ""},
    {"pattern": r"^\w+", "flags": "m"},
    {"pattern": r"\w+$", "flags": "m"},
    {"pattern": r"support", "flags": "i"},
    {"pattern": r"SUPPORT", "flags": ""},
    {"pattern": r"(\d{3})[-.](\d{4})", "flags": ""},
]
regex_cases = []
for case in REGEX:
    fl = (re.I if "i" in case["flags"] else 0) | (re.M if "m" in case["flags"] else 0)
    ms = [{"text": m.group(0), "start": m.start(), "end": m.end(), "groups": list(m.groups())}
          for m in re.finditer(case["pattern"], TEXT, fl)]
    regex_cases.append({**case, "matches": ms})
regex_replace = {"pattern": r"(\d{3})[-.](\d{4})", "replacement": "$2-$1",
                 "expected": re.sub(r"(\d{3})[-.](\d{4})", r"\2-\1", TEXT)}

# ---------------------------------------------------------------- random


class MT19937_64:
    """std::mt19937_64 (the C++ standard's parameters, seed(value))."""
    W = (1 << 64) - 1

    def __init__(self, seed):
        self.mt = [0] * 312
        self.mt[0] = seed & self.W
        for i in range(1, 312):
            self.mt[i] = (6364136223846793005 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 62)) + i) & self.W
        self.i = 312

    def __call__(self):
        if self.i >= 312:
            for k in range(312):
                y = (self.mt[k] & 0xFFFFFFFF80000000) | (self.mt[(k + 1) % 312] & 0x7FFFFFFF)
                v = self.mt[(k + 156) % 312] ^ (y >> 1)
                if y & 1:
                    v ^= 0xB5026F5AA96619E9
                self.mt[k] = v
            self.i = 0
        y = self.mt[self.i]
        self.i += 1
        y ^= (y >> 29) & 0x5555555555555555
        y ^= (y << 17) & 0x71D67FFFEDA60000
        y ^= (y << 37) & 0xFFF7EEE000000000
        y ^= y >> 43
        return y & self.W


def u01(r):
    return (r() >> 11) * 2.0 ** -53


def uniform(r, lo, hi):
    v = lo + (hi - lo) * u01(r)
    return v if v < hi else math.nextafter(hi, lo)


def normal(r, mean, sd):
    u1 = 1.0 - u01(r)
    u2 = u01(r)
    return mean + sd * math.sqrt(-2.0 * math.log(u1)) * math.cos(6.283185307179586476925286766559 * u2)


def exponential(r, lam):
    return -math.log(1.0 - u01(r)) / lam


def poisson(r, mean):
    if mean < 10.0:
        limit = math.exp(-mean)
        k = 0
        prod = 1.0 - u01(r)
        while prod > limit:
            k += 1
            prod *= 1.0 - u01(r)
        return k
    slam, loglam = math.sqrt(mean), math.log(mean)
    b = 0.931 + 2.53 * slam
    a = -0.059 + 0.02483 * b
    invalpha = 1.1239 + 1.1328 / (b - 3.4)
    vr = 0.9277 - 3.6224 / (b - 2.0)
    while True:
        u = u01(r) - 0.5
        v = u01(r)
        us = 0.5 - abs(u)
        k = math.floor((2.0 * a / us + b) * u + mean + 0.43)
        if us >= 0.07 and v <= vr:
            return int(k)
        if k < 0.0 or (us < 0.013 and v > us):
            continue
        if math.log(v) + math.log(invalpha) - math.log(a / (us * us) + b) <= -mean + k * loglam - math.lgamma(k + 1.0):
            return int(k)


def below(r, bound):
    mx = (1 << 64) - 1
    limit = mx - mx % bound
    v = r()
    while v >= limit:
        v = r()
    return v % bound


def histogram(values, integers):
    lo, hi = min(values), max(values)
    bins, first, width = 20, lo, (hi - lo) / 20
    if integers:
        n = hi - lo + 1.0
        width = 1.0 if n <= 50.0 else math.ceil(n / 20.0)
        bins = int(math.ceil(n / width))
        first = lo - 0.5
    elif not hi > lo:
        bins, first, width = 1, lo - 0.5, 1.0
    edges = [first + width * i for i in range(bins + 1)]
    counts = [0] * bins
    for v in values:
        counts[min(max(int(math.floor((v - first) / width)), 0), bins - 1)] += 1
    # numpy agrees wherever no value sits on an inner edge
    return edges, counts


RANDOM = [
    {"kind": "uniform", "count": 2000, "a": -2.0, "b": 5.0, "seed": 42},
    {"kind": "normal", "count": 3000, "a": 10.0, "b": 2.0, "seed": 7},
    {"kind": "exponential", "count": 2000, "a": 0.5, "b": 0.0, "seed": 11},
    {"kind": "poisson", "count": 2000, "a": 3.5, "b": 0.0, "seed": 3},
    {"kind": "poisson", "count": 2000, "a": 250.0, "b": 0.0, "seed": 5},
    {"kind": "integers", "count": 2000, "a": -5, "b": 12, "seed": 9},
    {"kind": "integers", "count": 2000, "a": 0, "b": 999, "seed": 13},
]
random_cases = []
for c in RANDOM:
    r = MT19937_64(c["seed"])
    if c["kind"] == "uniform":
        vals = [uniform(r, c["a"], c["b"]) for _ in range(c["count"])]
    elif c["kind"] == "normal":
        vals = [normal(r, c["a"], c["b"]) for _ in range(c["count"])]
    elif c["kind"] == "exponential":
        vals = [exponential(r, c["a"]) for _ in range(c["count"])]
    elif c["kind"] == "poisson":
        vals = [float(poisson(r, c["a"])) for _ in range(c["count"])]
    else:
        lo, hi = int(c["a"]), int(c["b"])
        vals = [float(lo + below(r, hi - lo + 1)) for _ in range(c["count"])]
    x = np.array(vals)
    integers = c["kind"] in ("poisson", "integers")
    edges, counts = histogram(vals, integers)
    np_counts, _ = np.histogram(x, bins=np.array(edges))
    assert list(np_counts) == counts, c
    random_cases.append({**c, "first": vals[:10], "mean": float(x.mean()), "std_population": float(x.std(ddof=0)),
                         "std_sample": float(x.std(ddof=1)), "q1": float(np.percentile(x, 25)),
                         "median": float(np.percentile(x, 50)), "q3": float(np.percentile(x, 75)),
                         "min": float(x.min()), "max": float(x.max()), "edges": edges, "counts": counts})

OUT.write_text(json.dumps({
    "hash": {"texts": hash_cases, "file": hash_file},
    "units": {"cases": unit_cases, "refusals": unit_refusals, "aliases": unit_aliases},
    "json": {"valid": json_valid, "invalid": json_invalid},
    "regex": {"text": TEXT, "cases": regex_cases, "replace": regex_replace},
    "random": random_cases,
}, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {OUT}: {len(unit_cases)} unit cases, {len(random_cases)} random cases")
