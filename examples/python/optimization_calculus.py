#!/usr/bin/env python3
"""Plans, portfolios, integrals and optimizer paths with pycyxwiz.optimize and pycyxwiz.calculus.

What it shows: the work behind Tools > Optimization and Calculus and the
Portfolio Weights node, from Python, each answer checked against scipy (and
torch.optim when it is installed):

1. a production plan (linear program) and what one more hour of each
   resource is worth (shadow prices);
2. the least risky mix of five assets from examples/cyxgraph/portfolio/
   returns.csv (120 months, made up for these examples), with and without a
   target return;
3. a probability as an integral, over a finite and an infinite range;
4. a derivative, exact and by differences, and the step where differences
   are most accurate;
5. Adam and SGD with momentum on the Rosenbrock valley.

Run it with a Python 3.12 that has numpy and scipy, naming the folder that
holds pycyxwiz (and, if different, the folder of its DLLs):

    python examples/python/optimization_calculus.py build/windows-release/lib/Release build/windows-release/bin/Release

Expected result:

    plan: 10 chairs, 40 tables, 5 desks earn 2500.00
      carpentry has 30.00 hours to spare; one more hour of finishing is worth 20.00, of packing 10.00; each desk ordered beyond 5 costs 10.00
    least risky mix: equity 0.0575, growth 0.0060, credit 0.0000, bonds 0.9365, gold 0.0000 (sd 1.0912 % a month)
    earning 0.50 % a month: equity 0.0847, growth 0.3125, credit 0.2029, bonds 0.4000, gold 0.0000 (sd 2.6530 % a month)
    P(-1 < Z < 1) = 0.682689492137086 (21 evaluations), over the whole line 1.000000000000000
    f'(1) of x^3 * exp(-x): exact 0.7357588823, central differences 0.7357588823 (best h about 1.8e-06)
    Rosenbrock from (-2, 2), 2000 steps: adam ends at (0.6455, 0.4158), sgd + momentum at (0.9999, 0.9998)
    every answer matches scipy

torch.optim's paths are checked too when torch is installed. Two things to
notice: the target return buys growth and credit and halves bonds, at more
than twice the risk; and on this narrow valley SGD with momentum reaches the
minimum (1, 1) in 2,000 steps while Adam at lr 0.05 needs about 5,000 (the
window's Optimizers view shows both paths on the valley's contours).

The portfolio answers are those of the Portfolio Weights nodes in
examples/cyxgraph/portfolio/portfolio.cyxgraph.
"""
import csv
import math
import os
import sys

folder = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.getcwd())
dll_folder = os.path.abspath(sys.argv[2] if len(sys.argv) > 2 else folder)
for d in (folder, dll_folder, r"C:\Program Files\ArrayFire\v3\lib"):
    if os.path.isdir(d) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(d)
sys.path.insert(0, folder)

import numpy as np  # noqa: E402
from scipy import integrate, optimize  # noqa: E402

import pycyxwiz.calculus as calc  # noqa: E402
import pycyxwiz.optimize as opt  # noqa: E402

problems = []


def agree(what, ours, theirs, tolerance):
    if np.max(np.abs(np.asarray(ours, float) - np.asarray(theirs, float))) > tolerance:
        problems.append("%s: %s vs scipy %s" % (what, ours, theirs))


# 1. A workshop makes chairs, tables and desks (profit 30, 50, 40 each) from
# carpentry, finishing and packing hours, and must make at least 5 desks.
c = [30, 50, 40]
rows = [([2, 4, 6], "<=", 240, "carpentry"), ([1, 2, 2], "<=", 100, "finishing"), ([1, 1, 1], "<=", 55, "packing"),
        ([0, 0, 1], ">=", 5, "desks ordered")]
plan = opt.linprog(c, rows, maximize=True)
print("plan: %.0f chairs, %.0f tables, %.0f desks earn %.2f" % (*plan["x"], plan["fun"]))
print("  carpentry has %.2f hours to spare; one more hour of finishing is worth %.2f, of packing %.2f;"
      " each desk ordered beyond 5 costs %.2f" % (plan["slack"][0], plan["shadow_prices"][1], plan["shadow_prices"][2],
                                                  -plan["shadow_prices"][3]))
reference = optimize.linprog([-v for v in c], A_ub=[[2, 4, 6], [1, 2, 2], [1, 1, 1], [0, 0, -1]], b_ub=[240, 100, 55, -5],
                             method="highs")
agree("plan", plan["x"], reference.x, 1e-7)
agree("plan value", plan["fun"], -reference.fun, 1e-7)
# scipy's marginals are d(its minimum)/d(b_ub): negated for a maximum, and
# once more for the desk row, written there as -desks <= -5.
agree("shadow prices", plan["shadow_prices"], [-m for m in reference.ineqlin.marginals[:3]] + [reference.ineqlin.marginals[3]], 1e-7)

# 2. The least risky mix of five assets, from 120 months of returns.
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "..", "cyxgraph", "portfolio", "returns.csv"), newline="") as f:
    table = list(csv.reader(f))
assets = table[0][1:]
returns = np.array([[float(v) for v in row[1:]] for row in table[1:]])
means, covariance = returns.mean(axis=0), np.cov(returns, rowvar=False)


def mix(target):
    r = opt.portfolio(means.tolist(), covariance.tolist(), target, long_only=True)
    # scipy's SLSQP needs the covariance scaled up (its tolerances are
    # absolute, as HiGHS's are; pycyxwiz scales for you).
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if target is not None:
        constraints.append({"type": "ineq", "fun": lambda w: means @ w - target})
    s = optimize.minimize(lambda w: 0.5 * w @ (covariance * 1e4) @ w, np.full(len(assets), 1 / len(assets)),
                          jac=lambda w: (covariance * 1e4) @ w, bounds=[(0, None)] * len(assets),
                          constraints=constraints, method="SLSQP", options={"ftol": 1e-15, "maxiter": 1000})
    agree("weights (target %s)" % target, r["x"], s.x, 1e-6)
    w = np.array(r["x"])
    return ", ".join("%s %.4f" % (a, abs(v)) for a, v in zip(assets, w)), 100 * math.sqrt(w @ covariance @ w)


print("least risky mix: %s (sd %.4f %% a month)" % mix(None))
print("earning 0.50 %% a month: %s (sd %.4f %% a month)" % mix(0.005))

# 3. The standard normal's probability within one sd, and over the line.
density = "exp(-x^2/2)/sqrt(2*pi)"
inside = calc.integrate(density, -1, 1)
whole = calc.integrate(density, -math.inf, math.inf)
print("P(-1 < Z < 1) = %.15f (%d evaluations), over the whole line %.15f" % (inside["value"], inside["evaluations"], whole["value"]))
agree("P(-1 < Z < 1)", inside["value"], integrate.quad(lambda x: math.exp(-x * x / 2) / math.sqrt(2 * math.pi), -1, 1)[0], 1e-14)
agree("P(-1 < Z < 1) = erf", inside["value"], math.erf(1 / math.sqrt(2)), 1e-14)
agree("the whole line", whole["value"], 1.0, 1e-12)

# 4. A derivative: exact from the formula, and by central differences.
d = calc.derivative("x^3 * exp(-x)", 1.0)
errors = calc.step_errors("x^3 * exp(-x)", 1.0)
best = errors["steps"][int(np.argmin(errors["central_error"]))]
print("f'(1) of x^3 * exp(-x): exact %.10f, central differences %.10f (best h about %.1e)"
      % (d["first"], d["numerical_first"], best))
agree("f'(1)", d["first"], 2 * math.exp(-1), 1e-15)

# 5. Two optimizers down the Rosenbrock valley (float32, as in training).
rosenbrock = "(1 - x)^2 + 100*(y - x^2)^2"
adam = opt.follow(rosenbrock, [-2, 2], "adam", lr=0.05, steps=2000)
sgd = opt.follow(rosenbrock, [-2, 2], "sgd", lr=0.001, momentum=0.9, steps=2000)
print("Rosenbrock from (-2, 2), 2000 steps: adam ends at (%.4f, %.4f), sgd + momentum at (%.4f, %.4f)"
      % (*adam["points"][-1], *sgd["points"][-1]))
try:
    import torch

    def torch_path(make, steps):
        p = torch.tensor([-2.0, 2.0], requires_grad=True)
        o = make([p])
        for _ in range(steps):
            o.zero_grad()
            x, y = p.detach().tolist()  # the exact gradient, as pycyxwiz feeds it
            p.grad = torch.tensor([-2 * (1 - x) - 400 * x * (y - x * x), 200 * (y - x * x)], dtype=torch.float32)
            o.step()
        return p.detach().tolist()

    agree("adam path", adam["points"][-1], torch_path(lambda q: torch.optim.Adam(q, lr=0.05), 2000), 2e-3)
    agree("sgd path", sgd["points"][-1], torch_path(lambda q: torch.optim.SGD(q, lr=0.001, momentum=0.9), 2000), 2e-3)
except ImportError:
    pass

if problems:
    print("\n".join(problems))
    sys.exit(1)
print("every answer matches scipy")
