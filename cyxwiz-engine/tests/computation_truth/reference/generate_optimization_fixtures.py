"""Truth for the optimization window's backend (TOFIX129 D10), for
test_computation_truth_optimization.

expression  cyxwiz::Expression: each formula's value and its exact partial
            derivatives at a few points, from sympy (the C++ text uses ^, the
            sympy text **).
quad        cyxwiz::Calculus::Integrate, adaptive: the exact value (sympy) and
            scipy.integrate.quad's value, error estimate and evaluations.
fixed       the fixed rules: numpy trapezoid / midpoint sums / scipy simpson /
            scipy fixed_quad on the same points, and numpy leggauss nodes.
derivative  exact f' and f'' (sympy) and the forward / central differences at
            the steps the error curve uses.
curvature   Rosenbrock's Hessian eigenvalues (numpy eigvalsh) at points, and
            the count of grid points with a negative one.

Run with sympy 1.14, numpy 2.3, scipy 1.17:
  python generate_optimization_fixtures.py
Writes ../fixtures/optimization_truth.json.
"""

import json
import os

import numpy as np
import sympy as sp
from scipy import integrate

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


QUAD = [  # C++ text, a, b, sympy text
    ("sin(x)", "0", "pi", "sin(x)"),
    ("exp(-x^2)", "-inf", "inf", "exp(-(x**2))"),
    ("1/(1 + x^2)", "0", "1", "1/(1 + x**2)"),
    ("sqrt(x)", "0", "1", "sqrt(x)"),
    ("x^2*exp(-x)", "0", "inf", "x**2*exp(-x)"),
    ("cos(50*x)", "0", "1", "cos(50*x)"),
    ("exp(x)", "1", "0", "exp(x)"),
    ("1/(1 + x^2)", "-inf", "0", "1/(1 + x**2)"),
    ("1/sqrt(x)", "0", "1", "1/sqrt(x)"),
    ("log(x)", "0", "1", "log(x)"),
]

BOUND = {"0": 0.0, "1": 1.0, "pi": np.pi, "inf": np.inf, "-inf": -np.inf}
SYM_BOUND = {"0": 0, "1": 1, "pi": sp.pi, "inf": sp.oo, "-inf": -sp.oo}


def numeric(text):
    f = sp.lambdify(x, sp.sympify(text, locals=SYMBOLS), "numpy")
    return lambda v: f(v)


def quad_cases():
    cases = []
    for text, a, b, sympy_text in QUAD:
        f = sp.sympify(sympy_text, locals=SYMBOLS)
        exact = number(sp.integrate(f, (x, SYM_BOUND[a], SYM_BOUND[b])))
        value, error, info = integrate.quad(numeric(sympy_text), BOUND[a], BOUND[b], full_output=1)
        cases.append({"text": text, "a": a, "b": b, "exact": exact, "quad": value,
                      "quad_error": error, "quad_evaluations": int(info["neval"])})
    return cases


def fixed_cases():
    f = np.sin
    a, b, n = 0.0, np.pi, 100
    xs = np.linspace(a, b, n + 1)
    mid = (xs[:-1] + xs[1:]) / 2
    rules = {
        "trapezoid": float(integrate.trapezoid(f(xs), xs)),
        "midpoint": float(np.sum(f(mid)) * (b - a) / n),
        "simpson": float(integrate.simpson(f(xs), x=xs)),
    }
    gauss = {str(k): float(integrate.fixed_quad(f, a, b, n=k)[0]) for k in (1, 2, 5, 10, 20)}
    nodes = {}
    for k in (1, 2, 5, 20, 64):
        t, w = np.polynomial.legendre.leggauss(k)
        nodes[str(k)] = {"nodes": t.tolist(), "weights": w.tolist()}
    return {"text": "sin(x)", "a": "0", "b": "pi", "n": n, "rules": rules, "gauss": gauss, "legendre": nodes}


DERIVATIVES = [("sin(x)", 1.0, "sin(x)"), ("exp(-x^2)", 0.5, "exp(-(x**2))"),
               ("x^3 - 2*x", -1.5, "x**3 - 2*x"), ("log(x)*sqrt(x)", 2.0, "log(x)*sqrt(x)")]


def derivative_cases():
    cases = []
    steps = [float(10.0 ** e) for e in np.linspace(-1, -6, 11)]
    for text, at, sympy_text in DERIVATIVES:
        f = sp.sympify(sympy_text, locals=SYMBOLS)
        d1, d2 = sp.diff(f, x), sp.diff(f, x, 2)
        g = sp.lambdify(x, f, "math")
        exact = number(d1.subs(x, sp.Float(at, 30)))
        forward = [abs((g(at + h) - g(at)) / h - exact) for h in steps]
        central = [abs((g(at + h) - g(at - h)) / (2 * h) - exact) for h in steps]
        cases.append({"text": text, "x": at, "first": exact, "second": number(d2.subs(x, sp.Float(at, 30))),
                      "steps": steps, "forward_error": forward, "central_error": central})
    return cases


def curvature_cases():
    def hessian(px, py):
        return np.array([[2 - 400 * py + 1200 * px * px, -400 * px], [-400 * px, 200.0]])
    points = [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [-1.0, 1.0], [0.5, -0.5]]
    at = [{"point": p, "eigenvalues": np.linalg.eigvalsh(hessian(*p)).tolist()} for p in points]
    negative = 0
    side = 40
    for j in range(side):
        for i in range(side):
            px, py = -2 + 4 * i / (side - 1), -1 + 4 * j / (side - 1)
            if np.linalg.eigvalsh(hessian(px, py))[0] < 0:
                negative += 1
    return {"text": "(1 - x)^2 + 100*(y - x^2)^2", "at": at,
            "range": {"x": [-2, 2], "y": [-1, 3], "per_side": side, "negative": negative}}


def main():
    fixture = {"expression": expression_cases(), "quad": quad_cases(), "fixed": fixed_cases(),
               "derivative": derivative_cases(), "curvature": curvature_cases()}
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(fixture, handle, indent=1)
        handle.write("\n")
    print("wrote", OUT, len(fixture["expression"]), "formulas")


if __name__ == "__main__":
    main()
