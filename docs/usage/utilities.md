# Utilities

**Tools > Utilities** opens one window with six tabs. Each menu item opens it
on its own tab:

| Tab | What it does | Menu item |
| --- | --- | --- |
| Calculator | Works out an expression, with your own variables and a history | Calculator |
| Units | Converts between units of length, mass, temperature, time, data, area, volume and speed | Unit Converter |
| Random | Draws seeded random values with their statistics and histogram; makes UUIDs | Random Generator |
| Hash | MD5, SHA-1, SHA-256 or SHA-512 of a text or a file; checks a hash you were given | Hash Generator |
| JSON | Checks, formats or minifies JSON, shows its tree, finds a value by path | JSON Viewer |
| Regex | Tests a regular expression, with its matches and a replacement | Regex Tester |

The same work is in Python as `pycyxwiz.utilities` (see [Python](#python) at
the end of this page). Every result on this page was checked against Python's
`hashlib`, `json` and `re`, numpy, and the exact unit definitions (test
`test_computation_truth_utilities`).

Above the tabs, **Save as my default** keeps the open tab, the digits shown,
the angle unit, the unit category, the hash algorithm, the JSON indent and the
regex flags for next time. The same defaults are in **Preferences > General >
Tools > Utilities**.

In a narrow (docked) window the controls wrap and each tab's two halves sit
one above the other.

## Calculator

Type an expression and press Enter or **Evaluate**: `2 * pi + sqrt(16)` gives
**10.28318531**, and under it the value in full (10.283185307179586) with
**Copy**.

- **Angles**: radians or degrees for `sin`, `cos`, `tan` and their inverses.
- **Digits**: how many significant digits the result shows (1 to 17).
- **Variables**: a name and a value (the value can be an expression); use the
  name in later expressions. **Remove** takes it out.
- **History**: the last 50 expressions; click one to work it out again.
- **Functions and constants**: `sin cos tan asin acos atan atan2 sinh cosh
  tanh sqrt cbrt exp log ln log10 log2 abs floor ceil round sign pow mod min
  max factorial`, `pi e phi`. It is the same parser as Tools > Optimization
  and Calculus and `pycyxwiz.calculus.Expression`.

## Units

Pick a category, type a number (`1.5`, `2e-6`), and pick the two units.
**Swap** exchanges them. The result shows with the conversion as a formula,
for example `ft = cm x 0.0328083989501312` or `°F = °C x 1.8 + 32`. The table
on the right shows the same value in every unit of the category.

Every factor is the exact definition:

- inch 0.0254 m, foot 0.3048 m, mile 1609.344 m, nautical mile 1852 m;
- pound 0.45359237 kg; tonne 1000 kg, short ton 2000 lb, long ton 2240 lb;
- km/h 1000/3600 m/s, knot 1852/3600 m/s;
- US gallon 231 cubic inches, imperial gallon 4.54609 L.

**Data units:** kB, MB and GB are powers of 1000, and KiB, MiB and GiB powers
of 1024. 1 GB = 0.9313225746 GiB.

**Temperatures** convert through Celsius with the defining constants
(°F = °C x 9/5 + 32, K = °C + 273.15), so 0 °C is exactly 32 °F and 100 °C
exactly 212 °F. A value below absolute zero is refused ("Below absolute zero
(-26.85 K)" for -300 °C); -459.67 °F is absolute zero.

**Names:** a unit is found by its id, label or alias in any case
("Kilometer", "metres", "qt", "fl_oz"), or by its symbol exactly as written:
"MB" is a megabyte, "Mb" is nothing (the megabit is "Mbit").

A year is 365 days.

The **Unit Converter** node uses the same table, so a column converted in a
graph gives the same numbers.

## Random

Pick a distribution and its parameters, the count (up to 1,000,000) and a
seed, then **Generate**:

| Distribution | Parameters |
| --- | --- |
| Uniform | Low, High |
| Normal | Mean, SD |
| Exponential | Rate |
| Poisson | Mean |
| Integers | Low, High |

**The same seed gives the same values on Windows, Linux and macOS** (see
[Seeds and reproducibility](seeds_and_reproducibility.md)). **New seed** picks
another.

The results area shows:

- the histogram, with its bin edges and empty bins drawn; whole numbers get
  one bin per value up to 50 values;
- the mean, median, both standard deviations ("sample, n − 1" and
  "population, n"), the quartiles (as numpy's `percentile`) and the range;
- the first values.

**Copy values**, **Save as CSV...** and **Send to Data Studio** take the values
further. Send to Data Studio adds a table such as `random_normal_seed42` with
one column, `value`.

**UUIDs (version 4)** come from the operating system's random source, 128 bits
each, never from the seed.

## Hash

Choose **Text** or **File** and the algorithm, then **Hash**:

- **Text** is hashed as UTF-8. The SHA-256 of `Hello, World!` is
  `dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f`.
- **A file** is read in 1 MB blocks in the background, with progress and
  **Stop**, so a file of any size can be hashed.

Paste a hash into **Check against**: it says **Match**, or where it differs
("Differs at character 7"). Case and spaces are ignored.

## JSON

Type or paste JSON (**Example** fills in a sample), then **Check**, **Format**
or **Minify**. **Indent** sets the spaces per level for Format.

- **Valid JSON** gives a summary ("2 objects · 1 array · 4 top-level keys ·
  depth 2 · 208 characters formatted · 157 minified"), the **Tree** (every
  level opens; keys stay in the document's order), and the formatted or
  minified text with **Copy** and **Use as input**.
- **Not valid** names the place where it went wrong, as Python's `json` does:
  "Line 4, column 3: unexpected string literal; expected '}'". It shows the
  line before, the line itself and a caret under the character.
- **Path**: type `config.max_workers` or `features[0]` (a leading `$.` also
  works) and press Enter to get that value, or the reason there is none
  (`No key "x"`, `[5] is past the end (2 items)`).

## Regex

Type a pattern and the text, then **Test** (or Enter):

- **Ignore case** makes letters match either case.
- **Multiline** makes `^` and `$` match at every line instead of only at the
  start and end of the text: `^\w+` finds 1 match without it, and the first
  word of every line with it.

The table lists each match with its line, column, its position in characters
(as Python's `re` counts them) and its groups. **Copy matches** copies them.
Under **Replace**, the replacement may use `$&` for the match and `$1` to `$9`
for the groups. **Replace all** shows the result with **Copy the result**.

The regex language is ECMAScript (C++ `std::regex`). `\w` and `\b` know ASCII
letters only: in `café`, `\w+` matches `caf`.

## Python

```python
import pycyxwiz as cx
u = cx.utilities

u.evaluate("2 * pi + sqrt(16)")                    # 10.283185307179586
u.convert(100, "km/h", "knot")                      # 53.99568034557235
u.conversion_formula("celsius", "fahrenheit")       # '°F = °C x 1.8 + 32'
u.hash("Hello, World!")                             # 'dffd6021...986f'
u.hash_file("data.bin", "md5", lambda done, total: print(done))   # only False stops
u.json_check(text)                                  # valid, formatted, minified, counts / line, column
u.json_get(text, "config.max_workers")              # the Python value (KeyError when missing)
u.regex_find(r"^\w+", text, multiline=True)         # text, start, end, line, column, groups
u.regex_replace(r"(\w+)@", text, "[$1]@")
u.random("normal", 1000, 10, 2, seed=42)            # values, mean, sd_sample, median, q1, q3, edges, counts
u.uuid4(5)
```

Errors raise: an expression that does not parse, an unknown unit, a bad
pattern, an unknown algorithm, text that is not JSON or a malformed JSON path
raises `ValueError`; a JSON path that is well formed but not in the document
raises `KeyError`. The example `examples/python/utilities.py` runs every tab's
work and prints the results.
