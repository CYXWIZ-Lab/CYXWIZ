---
title: Maths and optimization with pycyxwiz
description: Solve linear systems and decompose matrices with pycyxwiz.linalg, integrate and differentiate formulas with calculus, and solve linear and quadratic programs with optimize.
type: how-to
area: python
audience: user
status: experimental
since: 1.0.8
owner: python
last_reviewed: 2026-11-25
related:
  - reference/linalg.md
  - reference/calculus.md
  - reference/optimize.md
  - performance.md
  - ../main-window/optimization-and-calculus.md
---

Use this for numerical work in the style of MATLAB: `pycyxwiz.linalg` for
matrices, `pycyxwiz.calculus` for formulas you type, and
`pycyxwiz.optimize` for linear and quadratic programs. They are the code
behind the Engine's **Tools > Linear Algebra** and **Tools > Optimization
and Calculus**.

> Checked against pycyxwiz 0.2.0. The samples on this page run, in order,
> as one script.

## Linear algebra

Every function of `pycyxwiz.linalg` takes NumPy arrays and returns NumPy
arrays, computing in `float64`:

```python
import math
import numpy as np
import pycyxwiz as cx
import pycyxwiz.linalg as la

cx.initialize()

A = np.array([[3.0, 2.0], [1.0, 4.0]])
b = np.array([7.0, 9.0])
print(la.solve(A, b))                  # [1. 2.]
print(la.det(A), la.rank(A), la.trace(A))   # 10.0 2 7.0
print(la.norm(np.array([[3.0, 4.0], [0.0, 12.0]])))   # 13.0 (Frobenius)

U, S, Vt = la.svd(A)
Q, R = la.qr(A)
values, vectors = la.eig(np.array([[2.0, 0.0], [0.0, 3.0]]))
print(np.allclose(U @ np.diag(S) @ Vt, A), np.allclose(Q @ R, A), sorted(np.real(values).tolist()))   # True True [2.0, 3.0]
```

| Function | What it gives |
| --- | --- |
| `solve(A, b)`, `lstsq(A, b)` | x with Ax = b; the least-squares x for a tall A |
| `inv`, `transpose`, `matmul`, `det`, `rank`, `trace`, `norm`, `cond` | As named |
| `svd(A)`, `qr(A)`, `lu(A)`, `chol(A)`, `eig(A)` | The decompositions, as tuples; `eig` gives complex eigenvalues |
| `eye(n)`, `zeros(r, c)`, `ones(r, c)`, `diag(v)` | New matrices; `eye`, `zeros` and `ones` return lists |

Three kinds of input work, and the choice is about speed
([Performance](performance.md)):

```python
print(la.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]).tolist())   # lists work too: [[19.0, 22.0], [43.0, 50.0]]

At = cx.Tensor.from_numpy(A.astype(np.float32))
bt = cx.Tensor.from_numpy(b.astype(np.float32))
x = la.solve(At, bt)                   # tensors in, a tensor out, on the device
print(x.to_numpy().round(4).tolist())  # [1.0, 2.0]
```

The tensor forms exist for `solve`, `lstsq`, `matmul`, `inv`, `transpose`
and `norm`. In the Engine's Python session these functions are also there
without the prefix (`solve`, `svd`, `eye`, ...).

## Formulas: integrals, derivatives, curvature

`pycyxwiz.calculus` works on a formula typed as text, with `+ - * / ^`,
`pi`, `e`, and functions such as `sin`, `exp`, `log`, `sqrt` and `abs`.
Derivatives are exact, worked out from the formula:

```python
import pycyxwiz.calculus as calc

print(calc.integrate("sin(x)", 0, math.pi)["value"])                  # 1.9999999999999998
print(calc.integrate("exp(-x^2)", -math.inf, math.inf)["value"])      # 1.7724538509055163 (the square root of pi)
print(calc.integrate("sin(x)", 0, math.pi, rule="simpson", n=10)["value"])   # 2.0001095173150043

d = calc.derivative("x^3 - 2*x", -1.5)
print(d["first"], d["second"], d["first_formula"])   # 4.75 -9.0 3*x^2 - 2

f = calc.Expression("x^2 + 3*x")
print(f(2.0), f.derivative(), f.evaluate_many([[0.0, 1.0, 2.0]]))   # 10.0 2*x + 3 [0.0, 4.0, 10.0]

rosenbrock = "(1 - x)^2 + 100*(y - x^2)^2"
print(calc.curvature(rosenbrock, [0, 1])["eigenvalues"])   # [-398.0, 200.0]: a saddle at this point
```

`integrate` uses adaptive Gauss-Kronrod (as `scipy.integrate.quad`) unless
you ask for `trapezoid`, `midpoint`, `simpson` or `gauss`. A formula with a
name that is not a variable raises `ValueError`.

## Linear and quadratic programs

`pycyxwiz.optimize` solves linear programs (`linprog`) and convex
quadratic programs (`qp`) with HiGHS, the solver behind
`scipy.optimize.linprog`. A constraint is `(coefficients, "<=" | ">=" |
"=", right-hand side[, name])`:

```python
import pycyxwiz.optimize as opt

# Two products earn 3 and 2 a unit; machine hours x + y <= 4, labour x + 2y <= 6.
plan = opt.linprog([3, 2], [([1, 1], "<=", 4, "Machine"), ([1, 2], "<=", 6, "Labour")], maximize=True)
print(plan["status"], plan["x"], plan["fun"])          # optimal [4.0, 0.0] 12.0
print(plan["shadow_prices"], plan["binding"])          # [3.0, 0.0] [True, False]

print(opt.linprog([1, 1], [([1, 1], "<=", -1)])["status"])   # infeasible
```

The shadow price of a row is how much the best value changes when that
row's right-hand side grows by one: one more machine hour earns 3 more.
Bounds default to every x >= 0 for `linprog` and to free for `qp`.

`opt.portfolio(means, covariance, target_return=None)` finds the least risky
mix of assets, as the Portfolio Weights node; `opt.follow(formula, start,
"adam", lr=0.01)` gives the path a training optimizer takes on a formula.

## Related

- [pycyxwiz.linalg](reference/linalg.md),
  [pycyxwiz.calculus](reference/calculus.md) and
  [pycyxwiz.optimize](reference/optimize.md): every function and what it
  returns.
- [Optimization and calculus in the Engine](../main-window/optimization-and-calculus.md).
- [Examples](examples.md): `linear_algebra_calibration.py` and
  `optimization_calculus.py`, each answer checked against NumPy and SciPy.
