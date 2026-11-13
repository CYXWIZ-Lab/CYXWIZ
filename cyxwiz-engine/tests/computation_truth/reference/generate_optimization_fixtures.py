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
lp          scipy.optimize.linprog (HiGHS): x, the objective, each row's room,
            and shadow prices as d(objective)/d(rhs) by re-solving with the
            rhs raised by 1e-6 (no sign convention to get wrong); statuses
            of infeasible and unbounded problems.
qp          scipy.optimize.minimize SLSQP at tol 1e-14, shadow prices the same
            way.
paths       torch.optim in float32 (as training runs): the point after every
            100 steps, and the step at which a diverging run stops being
            finite. The gradient is the one FollowOptimizer uses: exact (sympy),
            in float64 at the float32 point, rounded to float32; float32
            autograd differs in the last bits, which Rosenbrock's valley grows
            to 0.4 % by step 800 (RMSprop), whatever the optimizer. Each
            checkpoint also holds the path's own sensitivity: how far torch
            moves when the start's y is one float32 ulp higher (RMSprop's path
            is chaotic past step 600: 5e-4 at 800, 9e-3 at 1000).
portfolio   the Portfolio Weights step: 120 periods of 4 assets' returns
            (seed 11, mock), the first 90 marked Train; the weights on the means and covariance (n - 1) of the Train rows and of all
            rows, with and without a target, long only and not: exact, by
            solving the KKT system of every active set (SLSQP's line search
            fails on these small variances).

Run with sympy 1.14, numpy 2.3, scipy 1.17, torch 2.10:
  python generate_optimization_fixtures.py
Writes ../fixtures/optimization_truth.json.
"""

import itertools
import json
import os

import numpy as np
import sympy as sp
import torch
from scipy import integrate, optimize

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


# (label, maximize, objective, rows [(name, coefficients, sense, rhs)], bounds or None)
LPS = [
    ("two products", True, [3, 2], [("C1", [1, 1], "<=", 4), ("C2", [1, 2], "<=", 6)], None),
    ("diet", False, [0.6, 0.35], [("protein", [5, 7], ">=", 8), ("fat", [4, 2], ">=", 15),
                                  ("fibre", [2, 1], ">=", 3)], None),
    ("production plan", True, [20, 30, 25, 15], [("Labour", [1, 2, 1, 1], "<=", 40),
                                                 ("Machine", [3, 1, 2, 1], "<=", 60),
                                                 ("Order", [1, 1, 1, 0], "=", 25)], None),
    ("bounds and a free variable", False, [1, -2, 1],
     [("R1", [1, 1, 1], "<=", 12), ("R2", [1, -1, 0], ">=", -4), ("R3", [0, 1, -1], "=", 1)],
     [[1, 3], [None, None], [-5, None]]),
    ("transport", False, [8, 6, 10, 9, 12, 13],
     [("S1", [1, 1, 1, 0, 0, 0], "<=", 50), ("S2", [0, 0, 0, 1, 1, 1], "<=", 60),
      ("D1", [1, 0, 0, 1, 0, 0], ">=", 30), ("D2", [0, 1, 0, 0, 1, 0], ">=", 40),
      ("D3", [0, 0, 1, 0, 0, 1], ">=", 35)], None),
    ("infeasible", False, [1, 1], [("C1", [1, 1], "<=", 1), ("C2", [1, 1], ">=", 3)], None),
    ("unbounded", True, [1, 1], [("C1", [1, -1], "<=", 1)], None),
]


def solve_lp(maximize, c, rows, bounds):
    a_ub, b_ub, a_eq, b_eq = [], [], [], []
    for _, a, sense, rhs in rows:
        if sense == "<=":
            a_ub.append(a); b_ub.append(rhs)
        elif sense == ">=":
            a_ub.append([-v for v in a]); b_ub.append(-rhs)
        else:
            a_eq.append(a); b_eq.append(rhs)
    sign = -1 if maximize else 1
    return optimize.linprog([sign * v for v in c], A_ub=a_ub or None, b_ub=b_ub or None, A_eq=a_eq or None,
                            b_eq=b_eq or None, bounds=bounds if bounds else (0, None), method="highs")


def lp_cases():
    cases = []
    for label, maximize, c, rows, bounds in LPS:
        r = solve_lp(maximize, c, rows, bounds)
        case = {"label": label, "maximize": maximize, "objective": c, "bounds": bounds,
                "rows": [{"name": n, "coefficients": a, "sense": s_, "rhs": b} for n, a, s_, b in rows],
                "status": {0: "optimal", 2: "infeasible", 3: "unbounded"}[r.status]}
        if r.status == 0:
            value = -r.fun if maximize else r.fun
            prices = []
            for i in range(len(rows)):
                bumped = [list(row) for row in rows]
                bumped[i][3] += 1e-6
                rb = solve_lp(maximize, c, bumped, bounds)
                prices.append(((-rb.fun if maximize else rb.fun) - value) / 1e-6)
            activity = [sum(ai * xi for ai, xi in zip(a, r.x)) for _, a, _, _ in rows]
            room = [max(0.0, (b - act) if s_ == "<=" else ((act - b) if s_ == ">=" else abs(act - b)))
                    for (_, _, s_, b), act in zip(rows, activity)]
            case.update({"x": r.x.tolist(), "value": value, "shadow_prices": prices, "slack": room})
        cases.append(case)
    return cases


# (label, Q, c, rows, bounds or None = free)
QPS = [
    ("two variables", [[2, 0], [0, 2]], [-2, -5], [("C1", [1, 1], "<=", 3)], None),
    ("portfolio", [[0.04, 0.006, 0.002], [0.006, 0.09, 0.009], [0.002, 0.009, 0.0225]], [0, 0, 0],
     [("budget", [1, 1, 1], "=", 1), ("return", [0.06, 0.12, 0.08], ">=", 0.08)], [[0, None]] * 3),
    ("box", [[4, 1], [1, 2]], [1, 1], [("R1", [1, 1], ">=", 1)], [[0, None], [0, None]]),
    ("no constraints", [[2, 1], [1, 2]], [-1, -1], [], None),
]


def solve_qp(Q, c, rows, bounds):
    Q = np.array(Q, dtype=float); c = np.array(c, dtype=float)
    cons = []
    for _, a, sense, rhs in rows:
        a = np.array(a, dtype=float)
        if sense == "<=":
            cons.append({"type": "ineq", "fun": lambda x, a=a, b=rhs: b - a @ x, "jac": lambda x, a=a: -a})
        elif sense == ">=":
            cons.append({"type": "ineq", "fun": lambda x, a=a, b=rhs: a @ x - b, "jac": lambda x, a=a: a})
        else:
            cons.append({"type": "eq", "fun": lambda x, a=a, b=rhs: a @ x - b, "jac": lambda x, a=a: a})
    x0 = np.full(len(c), 1.0 / len(c))
    r = optimize.minimize(lambda x: 0.5 * x @ Q @ x + c @ x, x0, jac=lambda x: Q @ x + c, constraints=cons,
                          bounds=bounds, method="SLSQP", options={"ftol": 1e-15, "maxiter": 1000})
    assert r.success, r.message
    return r


def qp_cases():
    cases = []
    for label, Q, c, rows, bounds in QPS:
        r = solve_qp(Q, c, rows, bounds)
        prices = []
        for i in range(len(rows)):
            bumped = [list(row) for row in rows]
            bumped[i][3] += 1e-5
            prices.append((solve_qp(Q, c, bumped, bounds).fun - r.fun) / 1e-5)
        cases.append({"label": label, "Q": Q, "c": c, "bounds": bounds,
                      "rows": [{"name": n, "coefficients": a, "sense": s_, "rhs": b} for n, a, s_, b in rows],
                      "x": r.x.tolist(), "value": float(r.fun), "shadow_prices": prices})
    return cases


ROSENBROCK = ("(1 - x)^2 + 100*(y - x^2)^2", "(1 - x)**2 + 100*(y - x**2)**2")
HIMMELBLAU = ("(x^2 + y - 11)^2 + (x + y^2 - 7)^2", "(x**2 + y - 11)**2 + (x + y**2 - 7)**2")

PATHS = [  # label, (C++ text, sympy text), start, optimizer, settings, steps
    ("sgd", ROSENBROCK, [-2.0, 2.0], "SGD", {"lr": 0.001}, 1000),
    ("momentum", ROSENBROCK, [-2.0, 2.0], "SGD", {"lr": 0.001, "momentum": 0.9}, 1000),
    ("adam", ROSENBROCK, [-2.0, 2.0], "Adam", {"lr": 0.01}, 1000),
    ("adamw", ROSENBROCK, [-2.0, 2.0], "AdamW", {"lr": 0.01}, 1000),
    ("nadam", ROSENBROCK, [-2.0, 2.0], "NAdam", {"lr": 0.01}, 1000),
    ("rmsprop", ROSENBROCK, [-2.0, 2.0], "RMSprop", {"lr": 0.01}, 1000),
    ("adagrad", ROSENBROCK, [-2.0, 2.0], "Adagrad", {"lr": 0.1}, 1000),
    ("himmelblau adam", HIMMELBLAU, [0.0, 0.0], "Adam", {"lr": 0.05}, 500),
    ("diverging sgd", ROSENBROCK, [-2.0, 2.0], "SGD", {"lr": 0.01}, 1000),
]


def follow(gradient, start, name, settings, steps):
    """torch's path: checkpoints every 100 steps and where it stops being finite."""
    p = torch.tensor(start, dtype=torch.float32)
    opt = getattr(torch.optim, name)([p], **settings)
    every, diverged = [], None
    for step in range(steps + 1):
        point = p.detach().tolist()
        if not all(np.isfinite(point)):
            diverged = step
            break
        if step % 100 == 0 or step == steps:
            every.append({"step": step, "point": point})
        if step == steps:
            break
        try:
            g = gradient(*[float(v) for v in point])
        except OverflowError:
            g = [float("inf"), float("inf")]
        p.grad = torch.tensor(g, dtype=torch.float32)
        opt.step()
    return every, diverged


def path_cases():
    cases = []
    for label, (text, sympy_text), start, name, settings, steps in PATHS:
        f = sp.sympify(sympy_text, locals=SYMBOLS)
        gradient = sp.lambdify((x, y), [sp.diff(f, x), sp.diff(f, y)], "math")
        every, diverged = follow(gradient, start, name, settings, steps)
        nudged = [start[0], float(np.nextafter(np.float32(start[1]), np.float32(np.inf)))]
        other, _ = follow(gradient, nudged, name, settings, steps)
        for e, o in zip(every, other):
            e["sensitivity"] = max(abs(a - b) for a, b in zip(e["point"], o["point"]))
        cases.append({"label": label, "text": text, "start": start, "optimizer": name, "settings": settings,
                      "steps": steps, "every": every, "diverged_at": diverged})
    return cases


def portfolio_case():
    rng = np.random.default_rng(11)
    mean = np.array([0.006, 0.012, 0.008, 0.004])
    vol = np.array([0.04, 0.09, 0.05, 0.02])
    corr = np.array([[1.0, 0.3, 0.2, 0.1], [0.3, 1.0, 0.4, 0.0], [0.2, 0.4, 1.0, 0.2], [0.1, 0.0, 0.2, 1.0]])
    cov = np.outer(vol, vol) * corr
    returns = rng.multivariate_normal(mean, cov, size=120)
    names = ["equity", "growth", "credit", "bonds"]
    partition = [0] * 90 + [1] * 15 + [2] * 15

    def weights(rows, target, long_only):
        """Exact: every active set of the convex QP (assets held at 0, the
        target met with equality or not) solved as a KKT system; the feasible
        solution with the least variance is the optimum."""
        mu = rows.mean(axis=0)
        sigma = np.cov(rows, rowvar=False)
        best = None
        n = len(mu)
        for held in itertools.product([False, True], repeat=n) if long_only else [(False,) * n]:
            free = [i for i in range(n) if not held[i]]
            if not free:
                continue
            for target_active in ([False, True] if target is not None else [False]):
                a = [np.ones(len(free))]
                b = [1.0]
                if target_active:
                    a.append(mu[free])
                    b.append(target)
                a = np.array(a)
                k = len(free)
                kkt = np.zeros((k + len(b), k + len(b)))
                kkt[:k, :k] = sigma[np.ix_(free, free)]
                kkt[:k, k:] = a.T
                kkt[k:, :k] = a
                try:
                    sol = np.linalg.solve(kkt, np.concatenate([np.zeros(k), b]))
                except np.linalg.LinAlgError:
                    continue
                w = np.zeros(n)
                w[free] = sol[:k]
                if long_only and (w < -1e-12).any():
                    continue
                if target is not None and mu @ w < target - 1e-12:
                    continue
                value = 0.5 * w @ sigma @ w
                if best is None or value < best[0] - 1e-18:
                    best = (value, w)
        w = best[1]
        share = w * (sigma @ w) / (w @ sigma @ w)
        return {"weight": w.tolist(), "mean_return": mu.tolist(), "sd": np.sqrt(np.diag(sigma)).tolist(),
                "risk_share": share.tolist()}

    train = returns[:90]
    # A target between the Train means' median and their best: reachable,
    # and binding (above the minimum-variance mix's return).
    means = train.mean(axis=0)
    reach = float(np.round(0.5 * (np.median(means) + means.max()), 5))
    cases = []
    for target in (None, reach):
        for long_only in (True, False):
            cases.append({"target_return": target, "long_only": long_only, "split": True,
                          **weights(train, target, long_only)})
    cases.append({"target_return": reach, "long_only": True, "split": False, **weights(returns, reach, True)})
    return {"assets": names, "returns": returns.tolist(), "partition": partition, "cases": cases,
            "unreachable": float(train.mean(axis=0).max() + 0.01)}


def main():
    fixture = {"expression": expression_cases(), "quad": quad_cases(), "fixed": fixed_cases(),
               "derivative": derivative_cases(), "curvature": curvature_cases(), "lp": lp_cases(),
               "qp": qp_cases(), "paths": path_cases(), "portfolio": portfolio_case()}
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump(fixture, handle, indent=1)
        handle.write("\n")
    print("wrote", OUT, len(fixture["expression"]), "formulas")


if __name__ == "__main__":
    main()
