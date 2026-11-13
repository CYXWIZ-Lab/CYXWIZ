"""pycyxwiz.utilities: the six tabs of Tools > Utilities from Python.

Nothing to download: a formula, a few unit conversions, the hash of a text
and of a file this script writes, a JSON document with a mistake in it, a
regex over a short contact list, seeded random values and UUIDs.

    python utilities.py <folder holding pycyxwiz> [<folder of its DLLs>]

The numbers it prints are in python_doc/pycyxwiz.md (6.6) and
docs/usage/utilities.md.
"""
import hashlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cyxwiz-backend", "python", "tests"))
from pycyxwiz_test_env import import_pycyxwiz  # noqa: E402

# The unit formulas print a degree sign; a Windows console may not take it.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

cx = import_pycyxwiz()
u = cx.utilities

print("== Calculator")
print("2 * pi + sqrt(16) =", u.evaluate("2 * pi + sqrt(16)"))
print("sin(30) in degrees =", u.evaluate("sin(30)", degrees=True))
print("rate * 1e4 with rate = 0.025:", u.evaluate("rate * 1e4", {"rate": 0.025}))

print("\n== Units (exact definitions)")
print("100 km/h =", u.convert(100, "km/h", "knot"), "knots")
print("1 GB =", u.convert(1, "GB", "GiB"), "GiB")
print("37 degC =", u.convert(37, "celsius", "fahrenheit"), "degF;", u.conversion_formula("celsius", "fahrenheit"))
print("1 tonne =", u.convert(1, "tonne", "short_ton"), "short tons")
try:
    u.convert(-300, "celsius", "kelvin")
except ValueError as e:
    print("-300 degC:", e)

print("\n== Hash")
print("SHA-256 of 'Hello, World!':", u.hash("Hello, World!"))
with tempfile.TemporaryDirectory() as folder:
    path = os.path.join(folder, "data.bin")
    data = bytes((i * 31 + 7) % 256 for i in range(3 * 1024 * 1024))
    with open(path, "wb") as out:
        out.write(data)
    blocks = []
    digest = u.hash_file(path, "md5", lambda done, total: blocks.append(done) or True)
    print(f"MD5 of a 3 MB file, read in {len(blocks)} blocks:", digest,
          "(hashlib agrees)" if digest == hashlib.md5(data).hexdigest() else "(hashlib DIFFERS)")

print("\n== JSON")
broken = '{\n  "name": "CyxWiz",\n  "version": "1.0.0"\n  "gpu": true\n}'
check = u.json_check(broken)
print(f"not valid: line {check['line']}, column {check['column']}")
fixed = broken.replace('"1.0.0"\n', '"1.0.0",\n')
check = u.json_check(fixed)
print("fixed:", check["minified"], f"({check['objects']} object, keys {check['keys']})")
print("gpu =", u.json_get(fixed, "gpu"))

print("\n== Regex (ECMAScript)")
text = "John Doe - john.doe@example.com\nJane Smith - jane@company.org\nSupport - help@help-desk.io"
for m in u.regex_find(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text):
    print(f"  {m['text']:22} line {m['line']}, column {m['column']}")
print("first word of each line:", [m["text"] for m in u.regex_find(r"^\w+", text, multiline=True)])
print(u.regex_replace(r"(\w+)@", text, "[$1]@").splitlines()[0])

print("\n== Random (seed 42: the same values on every platform)")
r = u.random("normal", 1000, 10, 2, seed=42)
print(f"Normal(10, 2), 1,000 values: mean {r['mean']:.4f}, SD (n - 1) {r['sd_sample']:.4f}, "
      f"median {r['median']:.4f}, Q1 {r['q1']:.4f}, Q3 {r['q3']:.4f}")
print("first values:", [round(v, 4) for v in r["values"][:5]])
dice = u.random("integers", 6000, 1, 6, seed=7)
print("6,000 dice throws, counts of 1..6:", dice["counts"])
print("UUIDs:", u.uuid4(2))
