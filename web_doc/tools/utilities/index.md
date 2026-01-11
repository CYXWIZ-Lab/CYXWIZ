# Utility Tools

CyxWiz Engine includes several utility tools for everyday tasks like calculations, conversions, and data inspection.

## Overview

Access utility tools via **Tools > Utilities** menu or the Command Palette (`Ctrl+Shift+P`).

| Tool | Description |
|------|-------------|
| [Calculator](#calculator) | Advanced scientific calculator |
| [Unit Converter](#unit-converter) | Physical unit conversions |
| [Hash Generator](#hash-generator) | Compute MD5, SHA1, SHA256 hashes |
| [Regex Tester](#regex-tester) | Test regular expressions |
| [JSON Viewer](#json-viewer) | View and validate JSON data |
| [Random Generator](#random-generator) | Generate random numbers and strings |

---

## Calculator

Advanced calculator with expression evaluation and history.

### Opening

- **Menu**: Tools > Utilities > Calculator
- **Command Palette**: "Open Calculator"

### Features

| Feature | Description |
|---------|-------------|
| **Expression Input** | Type mathematical expressions |
| **History** | View previous calculations |
| **Variables** | Store values for reuse |
| **Functions** | Built-in math functions |
| **Constants** | pi, e, phi |

### Basic Operations

```
5 + 3          → 8
10 - 4         → 6
6 * 7          → 42
20 / 4         → 5
2 ^ 8          → 256
17 % 5         → 2 (modulo)
```

### Built-in Functions

| Function | Description | Example |
|----------|-------------|---------|
| `sin(x)` | Sine (radians) | `sin(pi/2)` → 1 |
| `cos(x)` | Cosine | `cos(0)` → 1 |
| `tan(x)` | Tangent | `tan(pi/4)` → 1 |
| `asin(x)` | Arcsine | `asin(1)` → 1.571 |
| `acos(x)` | Arccosine | `acos(0)` → 1.571 |
| `atan(x)` | Arctangent | `atan(1)` → 0.785 |
| `sqrt(x)` | Square root | `sqrt(16)` → 4 |
| `cbrt(x)` | Cube root | `cbrt(27)` → 3 |
| `log(x)` | Natural log | `log(e)` → 1 |
| `log10(x)` | Base-10 log | `log10(100)` → 2 |
| `log2(x)` | Base-2 log | `log2(8)` → 3 |
| `exp(x)` | e^x | `exp(1)` → 2.718 |
| `abs(x)` | Absolute value | `abs(-5)` → 5 |
| `floor(x)` | Round down | `floor(3.7)` → 3 |
| `ceil(x)` | Round up | `ceil(3.2)` → 4 |
| `round(x)` | Round | `round(3.5)` → 4 |
| `factorial(x)` | Factorial | `factorial(5)` → 120 |

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `pi` | 3.14159... | Circle constant |
| `e` | 2.71828... | Euler's number |
| `phi` | 1.61803... | Golden ratio |
| `tau` | 6.28318... | 2π |

### Variables

Store and reuse values:

```
a = 5
b = 3
a + b        → 8
a * b        → 15
c = a ^ 2    → 25
sqrt(c)      → 5
```

### Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Evaluate | `Enter` |
| Previous | `Up Arrow` |
| Next | `Down Arrow` |
| Clear | `Escape` |
| Copy Result | `Ctrl+C` |

---

## Unit Converter

Convert between various physical units.

### Opening

- **Menu**: Tools > Utilities > Unit Converter
- **Command Palette**: "Open Unit Converter"

### Categories

#### Length

| Unit | Symbol |
|------|--------|
| Meters | m |
| Kilometers | km |
| Centimeters | cm |
| Millimeters | mm |
| Miles | mi |
| Yards | yd |
| Feet | ft |
| Inches | in |

**Examples:**
```
1 km → 1000 m
1 mi → 1.609 km
12 in → 30.48 cm
```

#### Mass

| Unit | Symbol |
|------|--------|
| Kilograms | kg |
| Grams | g |
| Milligrams | mg |
| Pounds | lb |
| Ounces | oz |
| Tonnes | t |

**Examples:**
```
1 kg → 2.205 lb
1 oz → 28.35 g
```

#### Temperature

| Unit | Symbol |
|------|--------|
| Celsius | °C |
| Fahrenheit | °F |
| Kelvin | K |

**Formulas:**
```
°F = °C × 9/5 + 32
K = °C + 273.15
```

#### Time

| Unit | Symbol |
|------|--------|
| Seconds | s |
| Milliseconds | ms |
| Microseconds | μs |
| Minutes | min |
| Hours | h |
| Days | d |
| Weeks | w |

#### Data Size

| Unit | Value |
|------|-------|
| Bytes | B |
| Kilobytes | KB (1024 B) |
| Megabytes | MB |
| Gigabytes | GB |
| Terabytes | TB |
| Bits | b |
| Kibibytes | KiB |

#### Speed

| Unit | Symbol |
|------|--------|
| Meters/second | m/s |
| Kilometers/hour | km/h |
| Miles/hour | mph |
| Knots | kn |
| Mach | M |

---

## Hash Generator

Compute cryptographic hashes of text or files.

### Opening

- **Menu**: Tools > Utilities > Hash Generator
- **Command Palette**: "Open Hash Generator"

### Supported Algorithms

| Algorithm | Output Length | Use Case |
|-----------|---------------|----------|
| MD5 | 128-bit (32 hex) | Checksums (not secure) |
| SHA-1 | 160-bit (40 hex) | Legacy systems |
| SHA-256 | 256-bit (64 hex) | Secure hashing |
| SHA-512 | 512-bit (128 hex) | High security |

### Usage

#### Hash Text

1. Select algorithm
2. Enter text in the input field
3. Hash is computed automatically
4. Click "Copy" to copy to clipboard

#### Hash File

1. Click "Open File"
2. Select file to hash
3. Hash is computed and displayed

### Example Outputs

```
Input: "Hello, World!"

MD5:    65a8e27d8879283831b664bd8b7f0ad4
SHA-1:  0a0a9f2a6772942557ab5355d76af442f8f65e01
SHA-256: dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f
```

---

## Regex Tester

Test and debug regular expressions.

### Opening

- **Menu**: Tools > Utilities > Regex Tester
- **Command Palette**: "Open Regex Tester"

### Features

| Feature | Description |
|---------|-------------|
| **Pattern Input** | Enter regex pattern |
| **Test String** | Text to match against |
| **Flags** | g (global), i (case-insensitive), m (multiline) |
| **Match Highlighting** | Visual match indication |
| **Capture Groups** | View captured groups |
| **Explanation** | Pattern breakdown |

### Usage

1. Enter your regex pattern
2. Set flags (optional)
3. Enter test string
4. Matches are highlighted in real-time

### Quick Reference

| Pattern | Matches |
|---------|---------|
| `.` | Any character |
| `\d` | Digit (0-9) |
| `\w` | Word character |
| `\s` | Whitespace |
| `^` | Start of string |
| `$` | End of string |
| `*` | 0 or more |
| `+` | 1 or more |
| `?` | 0 or 1 |
| `{n}` | Exactly n |
| `{n,m}` | n to m times |
| `[abc]` | Character class |
| `[^abc]` | Negated class |
| `(...)` | Capture group |
| `(?:...)` | Non-capture group |
| `\|` | Alternation |

### Example

**Pattern:** `\b\w+@\w+\.\w+\b`

**Test String:** `Contact us at info@example.com or support@test.org`

**Matches:**
- `info@example.com`
- `support@test.org`

---

## JSON Viewer

View, validate, and explore JSON data.

### Opening

- **Menu**: Tools > Utilities > JSON Viewer
- **Command Palette**: "Open JSON Viewer"

### Features

| Feature | Description |
|---------|-------------|
| **Tree View** | Expandable hierarchy |
| **Syntax Highlighting** | Color-coded JSON |
| **Validation** | Error detection |
| **Formatting** | Pretty print |
| **Minify** | Compact output |
| **Search** | Find keys/values |
| **Copy Path** | Get JSONPath |

### Usage

#### Load JSON

- Paste directly into the input
- Click "Open File" to load from file
- Drag and drop JSON file

#### Navigation

- Click arrows to expand/collapse
- Double-click to edit values
- Right-click for context menu

### Toolbar

| Button | Action |
|--------|--------|
| Format | Pretty print JSON |
| Minify | Remove whitespace |
| Validate | Check syntax |
| Copy | Copy to clipboard |
| Save | Save to file |

### Error Handling

Invalid JSON shows:
- Error message
- Line number
- Character position

```json
{
  "name": "test",
  "value": 123,  // ← Error: trailing comma
}
```

---

## Random Generator

Generate random numbers, strings, and data.

### Opening

- **Menu**: Tools > Utilities > Random Generator
- **Command Palette**: "Open Random Generator"

### Generator Types

#### Random Number

| Setting | Options |
|---------|---------|
| Type | Integer, Float |
| Min | Minimum value |
| Max | Maximum value |
| Count | How many to generate |
| Distribution | Uniform, Normal, Exponential |

**Example:**
```
Type: Integer
Min: 1, Max: 100
Count: 5
→ [42, 17, 89, 3, 56]
```

#### Random String

| Setting | Options |
|---------|---------|
| Length | String length |
| Characters | Uppercase, Lowercase, Digits, Symbols |
| Count | How many to generate |

**Example:**
```
Length: 16
Charset: A-Z, a-z, 0-9
→ "Kj7mXp2QwN8rYvLs"
```

#### UUID Generator

Generate UUIDs (Universally Unique Identifiers):

```
v4: f47ac10b-58cc-4372-a567-0e02b2c3d479
```

#### Password Generator

Generate secure passwords:

| Setting | Options |
|---------|---------|
| Length | 8-128 characters |
| Uppercase | Include A-Z |
| Lowercase | Include a-z |
| Numbers | Include 0-9 |
| Symbols | Include !@#$%^&* |
| Exclude | Ambiguous (0O, 1l) |

**Example:**
```
Length: 20
All options enabled
→ "Kj7#mXp2$QwN8!rYvLs@"
```

#### Data Generator

Generate structured test data:

| Type | Output |
|------|--------|
| Name | "John Smith" |
| Email | "john.smith@example.com" |
| Phone | "+1-555-123-4567" |
| Address | "123 Main St, City, State" |
| Date | "2025-03-15" |
| IP Address | "192.168.1.100" |
| Lorem Ipsum | Placeholder text |

### Copy & Export

- **Copy**: Copy generated values to clipboard
- **Export**: Save as CSV, JSON, or text file

---

## Keyboard Shortcuts

| Tool | Open Shortcut |
|------|---------------|
| Calculator | `Ctrl+Shift+C` |
| Unit Converter | `Ctrl+Shift+U` |
| Hash Generator | - |
| Regex Tester | `Ctrl+Shift+R` |
| JSON Viewer | `Ctrl+Shift+J` |
| Random Generator | - |

---

**Back to**: [Tools Index](../index.md)
