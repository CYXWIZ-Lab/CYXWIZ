"""Truth for the optimization window's backend (TOFIX129 D10), for
test_computation_truth_optimization.

expression  cyxwiz::Expression: each formula's value and its exact partial
            derivatives at a few points, from sympy (the C++ text uses ^, the
            sympy text **).

Run with sympy 1.14, numpy 2.3:
  python generate_optimization_fixtures.py
Writes ../fixtures/optimization_truth.json.
"""

import json
import os

import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "fixtures", "optimization_truth.json")

x, y = sp.symbols("x y", real=True)
SYMBOLS = {"x": x, "y": y}

# (C++ text, variables, sympy text)
FORMULAS = [
    ("(1 - x)^2 + 100*(y - x^2)^2", ["x", "y"], "(1 - x)**2 + 100*(y - x**2)**2"),
    ("-x^2", ["x"], "-(x**2)"),
    ("exp(-x^2)", ["x"], "exp(-(x**2))"),
    ("2^-1 + x^-2", ["x"], "2**(-1) + x**(-2)"),
    ("2^3^2", [], "2**(3**2)"),
    ("sin(x)*cos(y) + tan(x/3)", ["x", "y"], "sin(x)*cos(y) + tan(x/3)"),
    ("sqrt(x^2 + y^2)", ["x", "y"], "sqrt(x**2 + y**2)"),
    ("log(x) + ln(2*x) + log10(x) + log2(x) + cbrt(x)", ["x"], "log(x) + log(2*x) + log(x, 10) + log(x, 2) + x**(sp.Rational(1, 3))"),
    ("atan2(y, x)", ["x", "y"], "atan2(y, x)"),
    ("x^y", ["x", "y"], "x**y"),
    ("pow(x, 3) - 2*x", ["x"], "x**3 - 2*x"),
    ("asin(x/3) + acos(x/4) + atan(x)", ["x"], "asin(x/3) + acos(x/4) + atan(x)"),
    ("sinh(x) + cosh(x)*tanh(y)", ["x", "y"], "sinh(x) + cosh(x)*tanh(y)"),
    ("abs(x - 1)*exp(y)", ["x", "y"], "Abs(x - 1)*exp(y)"),
    ("1/(1 + x^2)", ["x"], "1/(1 + x**2)"),
    ("2e-3*x + .5 - 1.5E+2", ["x"], "sp.Rational(2, 1000)*x + sp.Rational(1, 2) - 150"),
    ("3*(x + 2)^3 - 4*x/y", ["x", "y"], "3*(x + 2)**3 - 4*x/y"),
    ("e^x + pi*phi + 2*E", ["x"], "exp(x) + pi*(1 + sqrt(5))/2 + 2*E"),
    ("x - (y - (x - y))", ["x", "y"], "x - (y - (x - y))"),
    ("x/(y/x)/2", ["x", "y"], "x/(y/x)/2"),
    ("(x + y)^(x - y)", ["x", "y"], "(x + y)**(x - y)"),
]

POINTS = [[0.7, 1.3], [1.9, 0.4], [2.6, 2.2]]


def number(value):
    v = complex(sp.N(value, 30))
    assert abs(v.imag) < 1e-25, value
    return float(v.real)


def expression_cases():
    cases = []
    for text, variables, sympy_text in FORMULAS:
        f = sp.sympify(sympy_text, locals={**SYMBOLS, "sp": sp, "E": sp.E})
        symbols = [SYMBOLS[v] for v in variables]
        points, values, derivatives = [], [], []
        for p in POINTS:
            point = p[: len(variables)]
            subs = dict(zip(symbols, [sp.Float(v, 30) for v in point]))
            points.append(point)
            values.append(number(f.subs(subs)))
            derivatives.append([number(sp.diff(f, s).subs(subs)) for s in symbols])
        cases.append({"text": text, "variables": variables, "points": points,
                      "values": values, "derivatives": derivatives})
    return cases


def main():
    fixture = {"expression": expression_cases()}
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(fixture, handle, indent=1)
        handle.write("\n")
    print("wrote", OUT, len(fixture["expression"]), "formulas")


if __name__ == "__main__":
    main()
