# SPDX-License-Identifier: Apache-2.0
# Part of the CyxWiz plugin SDK examples (see cmake/sdk/NOTICE).
# TOFIX129 D10.
"""pycyxwiz.optimize and pycyxwiz.calculus against the Python truth.

The cases are cyxwiz-engine/tests/computation_truth/fixtures/
optimization_truth.json, written by generate_optimization_fixtures.py from
sympy (values and exact derivatives), scipy (quad, the fixed rules, linprog,
SLSQP), numpy (Gauss-Legendre, eigenvalues), torch.optim (the optimizer
paths) and an exact KKT solve (the portfolio). The tolerances are those of
test_computation_truth_optimization. Run with the Python that pycyxwiz was
built for:

    python test_python_optimize.py <folder holding pycyxwiz> [<folder of its DLLs>]

Exit code 0 when every check passes.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pycyxwiz_test_env import Checks, import_pycyxwiz  # noqa: E402

cyx = import_pycyxwiz()
check = Checks()
calc, opt = cyx.calculus, cyx.optimize

here = os.path.dirname(os.path.abspath(__file__))
fixture = os.path.join(here, "..", "..", "..", "cyxwiz-engine", "tests", "computation_truth", "fixtures", "optimization_truth.json")
with open(fixture, encoding="utf-8") as f:
    truth = json.load(f)


def near(a, b, tolerance):
    return abs(a - b) <= tolerance * (1.0 + abs(b))


def bound(text):
    return {"inf": math.inf, "-inf": -math.inf, "pi": math.pi}.get(text, None) or float(text)


def raises(call, words):
    try:
        call()
    except ValueError as e:
        return words in str(e)
    return False


# Expressions: values and exact derivatives at the fixture's points (sympy).
worst = 0.0
for c in truth["expression"]:
    f = calc.Expression(c["text"], c["variables"])
    again = calc.Expression(str(f), c["variables"])
    partials = [f.derivative(v) for v in c["variables"]]
    for p, point in enumerate(c["points"]):
        worst = max(worst, abs(f(*point) - c["values"][p]) / (1 + abs(c["values"][p])),
                    abs(again(*point) - c["values"][p]) / (1 + abs(c["values"][p])))
        for v, d in enumerate(partials):
            worst = max(worst, abs(d(*point) - c["derivatives"][p][v]) / (10 + 10 * abs(c["derivatives"][p][v])))
    if not c["variables"]:
        continue  # a constant: no columns to evaluate over
    many = f.evaluate_many([list(col) for col in zip(*c["points"])])
    check(len(many) == len(c["points"]), "evaluate_many %s: one value per point" % c["text"])
    worst = max(worst, *(abs(many[p] - c["values"][p]) / (1 + abs(c["values"][p])) for p in range(len(c["points"]))))
check(worst <= 1e-12, "Expression: %d formulas, values, printed and re-read, exact partials, evaluate_many (largest %.1e)"
      % (len(truth["expression"]), worst))
check(calc.Expression("-2^2", [])() == -4.0 and abs(calc.Expression("sin(30)", [], degrees=True)() - 0.5) < 1e-15,
      "Expression: -2^2 is -(2^2); degrees")
check(raises(lambda: calc.Expression("x + z"), "Unknown name 'z'"), "Expression refuses an unknown name")
check(raises(lambda: calc.Expression("1/x")(0.0), "Division by zero"), "Expression: a division by zero raises")
check(str(calc.Expression("x^3", ["x"]).derivative()) == "3*x^2", "derivative() prints simplified: 3*x^2")

# Integrals: adaptive against quad, the fixed rules against scipy.
for c in truth["quad"]:
    label = "%s from %s to %s" % (c["text"], c["a"], c["b"])
    r = calc.integrate(c["text"], bound(c["a"]), bound(c["b"]))
    miss = abs(r["value"] - c["exact"])
    ok = miss <= max(r["error"], 4e-16 * (1 + abs(c["exact"])))
    if r["converged"]:
        ok = ok and miss <= max(1.49e-8, 1.49e-8 * abs(c["exact"]))
    if c["quad_evaluations"] == 21:
        ok = ok and r["evaluations"] == 21 and abs(r["value"] - c["quad"]) <= 1e-15 * (1 + abs(c["quad"]))
    check(ok, "integrate %s: %.17g (estimate %.1e%s)" % (label, r["value"], r["error"], "" if r["converged"] else ", not converged"))
fx = truth["fixed"]
a, b = bound(fx["a"]), bound(fx["b"])
for rule, value in fx["rules"].items():
    r = calc.integrate(fx["text"], a, b, rule=rule, n=fx["n"])
    check(near(r["value"], value, 1e-13) and r["error"] is None, "integrate rule=%s n=%d: %.17g" % (rule, fx["n"], r["value"]))
for n, value in fx["gauss"].items():
    check(near(calc.integrate(fx["text"], a, b, rule="gauss", n=int(n))["value"], value, 1e-14), "integrate rule=gauss n=%s" % n)
for n, nw in fx["legendre"].items():
    nodes, weights = calc.gauss_legendre(int(n))
    check(all(near(x, e, 1e-14) for x, e in zip(nodes, nw["nodes"])) and all(near(w, e, 1e-13) for w, e in zip(weights, nw["weights"])),
          "gauss_legendre(%s) = numpy leggauss" % n)
check(raises(lambda: calc.integrate("x", 0, 1, rule="simpson", n=7), "even number"), "integrate refuses Simpson with an odd n")
check(raises(lambda: calc.integrate("exp(-x)", 0, math.inf, rule="trapezoid", n=10), "adaptive"), "infinite bounds need adaptive")

# Derivatives: exact against sympy, the step errors against numpy.
for c in truth["derivative"]:
    r = calc.derivative(c["text"], c["x"])
    curve = calc.step_errors(c["text"], c["x"], h_from=c["steps"][0], h_to=c["steps"][-1], count=len(c["steps"]))
    fx = abs(calc.Expression(c["text"])(c["x"])) + 1.0
    steps_ok = all(near(curve["steps"][i], h, 1e-12)
                   and abs(curve["forward_error"][i] - c["forward_error"][i]) <= max(1e-3 * c["forward_error"][i], 8 * 2.2e-16 * fx / h)
                   and abs(curve["central_error"][i] - c["central_error"][i]) <= max(1e-3 * c["central_error"][i], 8 * 2.2e-16 * fx / h)
                   for i, h in enumerate(c["steps"]))
    check(near(r["first"], c["first"], 1e-14) and near(r["second"], c["second"], 1e-13)
          and abs(r["numerical_first"] - r["first"]) <= 1e-9 * (1 + abs(r["first"])) and steps_ok,
          "derivative and step_errors of %s at %g" % (c["text"], c["x"]))
none = calc.derivative("max(x, 0)", 1.0)
check(none["first"] is None and abs(none["numerical_first"] - 1) < 1e-9, "derivative: no exact one for max, numerical still")

# Curvature: eigenvalues against numpy, the range count.
cv = truth["curvature"]
for p in cv["at"]:
    r = calc.curvature(cv["text"], p["point"])
    check(all(near(x, e, 1e-10) for x, e in zip(r["eigenvalues"], p["eigenvalues"])), "curvature at %s" % p["point"])
rg = cv["range"]
r = calc.curvature_over_range(cv["text"], tuple(rg["x"]), tuple(rg["y"]), per_side=rg["per_side"])
check(r["points"] == 1600 and r["negative_points"] == rg["negative"], "curvature_over_range: %d of 1600 points curve down"
      % r["negative_points"])


def constraints(rows):
    return [(r["coefficients"], r["sense"], r["rhs"], r["name"]) for r in rows]


def program_ok(r, c, price_tolerance):
    return (r["success"] and len(r["x"]) == len(c["x"]) and all(near(x, e, 1e-7) for x, e in zip(r["x"], c["x"]))
            and near(r["fun"], c["value"], 1e-9)
            and len(r["shadow_prices"]) == len(c["shadow_prices"])
            and all(near(p, e, price_tolerance) for p, e in zip(r["shadow_prices"], c["shadow_prices"])))


# Programs against scipy.
for c in truth["lp"]:
    r = opt.linprog(c["objective"], constraints(c["rows"]), c["bounds"], maximize=c["maximize"])
    if c["status"] != "optimal":
        check(not r["success"] and r["status"] == c["status"] and r["fun"] is None, "linprog %s: %s" % (c["label"], r["message"]))
        continue
    check(program_ok(r, c, 1e-6) and all(near(s, e, 1e-7) for s, e in zip(r["slack"], c["slack"]))
          and r["binding"] == [s < 1e-9 for s in c["slack"]], "linprog %s: %s" % (c["label"], r["x"]))
for c in truth["qp"]:
    r = opt.qp(c["Q"], c["c"], constraints(c["rows"]), c["bounds"])
    # Shadow prices: scipy's SLSQP has none, the fixture's come by re-solving.
    check(program_ok(r, c, 2e-4), "qp %s: %s" % (c["label"], r["x"]))
r = opt.qp([[1, 2], [2, 1]], [0, 0])
check(r["status"] == "not convex" and "eigenvalue" in r["message"], "qp refuses a Q that is not convex: " + r["message"])
check(raises(lambda: opt.linprog([1], [([1], "<", 1)]), "'<'"), "linprog refuses an unknown sense")
check(abs(opt.linprog([-1, -1], bounds=(0, 0.5))["fun"] + 1.0) < 1e-9, "linprog: one (low, high) pair for every variable")
check(abs(opt.linprog([-1, -1], bounds=[(0, 1), (0, 2)])["fun"] + 3.0) < 1e-9, "linprog: a (low, high) pair per variable")
try:
    opt.linprog([1, 1], bounds=5)
    check(False, "linprog: bounds of the wrong type raise TypeError")
except TypeError:
    check(True, "linprog: bounds of the wrong type raise TypeError")
check(raises(lambda: calc.Expression(".e5"), "Expected a number") and raises(lambda: calc.Expression("1e400"), "out of range"),
      "Expression: '.e5' and '1e400' are refused in words")
check(str(calc.Expression("2^(-x*y)", ["x", "y"])) == "2^(-x*y)", "Expression prints the brackets a negated exponent needs")
check(raises(lambda: opt.follow("x^2", [1.0], "sgd", momentum=-0.1), "refused its settings"),
      "follow: an optimizer that refuses its settings is a ValueError")
ragged = opt.qp([[1, 2], []], [0, 0])
check(ragged["status"] == "failed" and "square" in ragged["message"], "qp: a ragged Q is refused: " + ragged["message"])
check(calc.curvature_over_range("(0.1*x + 0.3*y)^2", (-1, 1), (-1, 1))["negative_points"] == 0,
      "curvature_over_range: a semidefinite Hessian does not curve down")


# The portfolio against the exact KKT solve: means and covariance (n - 1) here.
def moments(rows):
    n, k = len(rows), len(rows[0])
    mean = [sum(r[j] for r in rows) / n for j in range(k)]
    cov = [[sum((r[i] - mean[i]) * (r[j] - mean[j]) for r in rows) / (n - 1) for j in range(k)] for i in range(k)]
    return mean, cov


pf = truth["portfolio"]
for c in pf["cases"]:
    rows = [r for r, part in zip(pf["returns"], pf["partition"]) if part == 0] if c["split"] else pf["returns"]
    mean, cov = moments(rows)
    r = opt.portfolio(mean, cov, c["target_return"], c["long_only"])
    check(r["success"] and all(abs(w - e) <= 1e-7 + 1e-7 * abs(e) for w, e in zip(r["x"], c["weight"])),
          "portfolio (%s, %s, target %s): %s" % ("Train rows" if c["split"] else "every row",
                                                 "long only" if c["long_only"] else "long and short", c["target_return"],
                                                 [round(w, 6) for w in r["x"]]))
mean, cov = moments(pf["returns"])
check(opt.portfolio(mean, cov, pf["unreachable"])["status"] == "infeasible", "portfolio: an unreachable target is infeasible")

# Optimizer paths against torch.optim (float32, fed the same exact gradient).
for c in truth["paths"]:
    s = c["settings"]
    path = opt.follow(c["text"], c["start"], c["optimizer"], lr=s["lr"], momentum=s.get("momentum", 0.0),
                      steps=c["steps"], tolerance=0.0)
    if c["diverged_at"] is not None:
        check(path["diverged"] and path["diverged_at"] == c["diverged_at"], "follow %s diverges at step %d" % (c["label"], c["diverged_at"]))
        continue
    ok = not path["diverged"] and len(path["points"]) == c["steps"] + 1
    for e in c["every"]:
        got = path["points"][e["step"]]
        ok = ok and all(abs(got[i] - e["point"][i]) <= max(2e-4 * (1 + abs(e["point"][i])), 4 * e["sensitivity"]) for i in range(2))
    check(ok, "follow %s: %d steps, at the end %s (torch %s)" % (c["label"], c["steps"], [round(v, 4) for v in path["points"][-1]],
                                                               [round(v, 4) for v in c["every"][-1]["point"]]))
flat = opt.follow("x^2", [3.0], "sgd", lr=0.1, tolerance=1e-6)
check(flat["converged"] and len(flat["points"]) < 100, "follow ends when the gradient is small")
check(raises(lambda: opt.follow("x^2", [1.0], "lbfgs"), "optimizer is one of"), "follow refuses an unknown optimizer")

check.finish("pycyxwiz.optimize / calculus")
