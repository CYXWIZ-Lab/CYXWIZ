# Utilities Quick Reference

This page provides a condensed reference for all utility tools in CyxWiz.

## Calculator Quick Reference

### Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `+` | Addition | `5 + 3` → 8 |
| `-` | Subtraction | `10 - 4` → 6 |
| `*` | Multiplication | `6 * 7` → 42 |
| `/` | Division | `20 / 4` → 5 |
| `^` | Power | `2 ^ 8` → 256 |
| `%` | Modulo | `17 % 5` → 2 |
| `()` | Parentheses | `(2 + 3) * 4` → 20 |

### Functions

```
sin(x), cos(x), tan(x)       Trigonometric
asin(x), acos(x), atan(x)    Inverse trig
sinh(x), cosh(x), tanh(x)    Hyperbolic
sqrt(x), cbrt(x)             Roots
log(x), log10(x), log2(x)    Logarithms
exp(x)                        Exponential
abs(x), floor(x), ceil(x)    Rounding
factorial(n)                  n!
min(a, b), max(a, b)         Min/max
```

### Constants

```
pi    = 3.14159265...
e     = 2.71828182...
phi   = 1.61803398...
tau   = 6.28318530...
```

---

## Unit Conversion Tables

### Length

| From | To | Multiply by |
|------|----|-------------|
| meters | feet | 3.28084 |
| meters | inches | 39.3701 |
| kilometers | miles | 0.621371 |
| inches | centimeters | 2.54 |
| feet | meters | 0.3048 |
| miles | kilometers | 1.60934 |

### Mass

| From | To | Multiply by |
|------|----|-------------|
| kilograms | pounds | 2.20462 |
| kilograms | ounces | 35.274 |
| pounds | kilograms | 0.453592 |
| grams | ounces | 0.035274 |

### Temperature

| From | To | Formula |
|------|----|---------|
| °C | °F | F = C × 9/5 + 32 |
| °F | °C | C = (F - 32) × 5/9 |
| °C | K | K = C + 273.15 |
| K | °C | C = K - 273.15 |

### Data Size

| Unit | Bytes |
|------|-------|
| 1 KB | 1,024 |
| 1 MB | 1,048,576 |
| 1 GB | 1,073,741,824 |
| 1 TB | 1,099,511,627,776 |

### Time

| From | To | Value |
|------|----|-------|
| 1 minute | seconds | 60 |
| 1 hour | seconds | 3,600 |
| 1 day | seconds | 86,400 |
| 1 week | seconds | 604,800 |

---

## Hash Algorithms Comparison

| Algorithm | Bits | Hex Chars | Speed | Security |
|-----------|------|-----------|-------|----------|
| MD5 | 128 | 32 | Fast | Broken |
| SHA-1 | 160 | 40 | Fast | Weak |
| SHA-256 | 256 | 64 | Medium | Strong |
| SHA-512 | 512 | 128 | Medium | Strong |

### Use Cases

- **MD5**: File checksums (non-security)
- **SHA-256**: Digital signatures, passwords
- **SHA-512**: High-security applications

---

## Regex Quick Reference

### Character Classes

| Pattern | Matches |
|---------|---------|
| `.` | Any character except newline |
| `\d` | Digit [0-9] |
| `\D` | Non-digit |
| `\w` | Word character [a-zA-Z0-9_] |
| `\W` | Non-word character |
| `\s` | Whitespace |
| `\S` | Non-whitespace |

### Anchors

| Pattern | Position |
|---------|----------|
| `^` | Start of string |
| `$` | End of string |
| `\b` | Word boundary |
| `\B` | Non-word boundary |

### Quantifiers

| Pattern | Meaning |
|---------|---------|
| `*` | 0 or more |
| `+` | 1 or more |
| `?` | 0 or 1 |
| `{n}` | Exactly n |
| `{n,}` | n or more |
| `{n,m}` | Between n and m |

### Groups

| Pattern | Description |
|---------|-------------|
| `(...)` | Capturing group |
| `(?:...)` | Non-capturing group |
| `(?=...)` | Positive lookahead |
| `(?!...)` | Negative lookahead |
| `(?<=...)` | Positive lookbehind |
| `(?<!...)` | Negative lookbehind |

### Common Patterns

```regex
# Email
\b[\w._%+-]+@[\w.-]+\.[a-zA-Z]{2,}\b

# URL
https?://[\w.-]+(?:/[\w./-]*)?

# Phone (US)
\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}

# IP Address
\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b

# Date (YYYY-MM-DD)
\d{4}-\d{2}-\d{2}

# Time (HH:MM:SS)
\d{2}:\d{2}:\d{2}

# Hex Color
#[0-9A-Fa-f]{6}

# Credit Card
\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}
```

---

## JSON Path Reference

### Syntax

| Expression | Description |
|------------|-------------|
| `$` | Root object |
| `.property` | Child property |
| `['property']` | Child property (bracket) |
| `[n]` | Array index |
| `[*]` | All elements |
| `..property` | Recursive descent |
| `[start:end]` | Array slice |

### Examples

```json
{
  "store": {
    "books": [
      {"title": "Book 1", "price": 10},
      {"title": "Book 2", "price": 20}
    ],
    "name": "My Store"
  }
}
```

| Path | Result |
|------|--------|
| `$.store.name` | "My Store" |
| `$.store.books[0].title` | "Book 1" |
| `$.store.books[*].price` | [10, 20] |
| `$..title` | ["Book 1", "Book 2"] |

---

## Random Generator Formulas

### Uniform Distribution

```python
value = min + random() * (max - min)
```

### Normal Distribution (Box-Muller)

```python
u1 = random()
u2 = random()
z = sqrt(-2 * log(u1)) * cos(2 * pi * u2)
value = mean + z * std_dev
```

### UUID v4 Format

```
xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx

Where:
- x = random hex digit
- 4 = version 4
- y = 8, 9, a, or b
```

### Password Strength

| Length | Charset | Entropy (bits) |
|--------|---------|----------------|
| 8 | a-z | ~38 |
| 12 | a-zA-Z0-9 | ~71 |
| 16 | all | ~105 |
| 20 | all | ~131 |

**Recommendation:** Minimum 12 characters with mixed case, numbers, and symbols (~80+ bits).

---

## Keyboard Shortcuts Summary

| Action | Shortcut |
|--------|----------|
| Open Calculator | `Ctrl+Shift+C` |
| Open Unit Converter | `Ctrl+Shift+U` |
| Open Regex Tester | `Ctrl+Shift+R` |
| Open JSON Viewer | `Ctrl+Shift+J` |
| Copy Result | `Ctrl+C` |
| Paste | `Ctrl+V` |
| Clear Input | `Escape` |
| Evaluate/Apply | `Enter` |

---

## Tips & Best Practices

### Calculator

1. Use parentheses for complex expressions
2. Assign intermediate results to variables
3. Use `ans` to reference last result

### Unit Converter

1. Double-check temperature conversions
2. Be aware of metric vs imperial
3. Note that KB = 1024 bytes (binary), not 1000

### Hash Generator

1. Never store passwords as MD5/SHA-1
2. Use SHA-256 minimum for security
3. Include salt for password hashing

### Regex Tester

1. Start simple, add complexity gradually
2. Test edge cases
3. Consider case sensitivity
4. Watch for backtracking in complex patterns

### JSON Viewer

1. Validate before processing
2. Use tree view for deep structures
3. Search by key name for large files

### Random Generator

1. Use cryptographic random for security
2. Avoid predictable seeds in production
3. Generate more than needed for statistical tests

---

**Back to**: [Utilities Overview](index.md) | [Tools Index](../index.md)
