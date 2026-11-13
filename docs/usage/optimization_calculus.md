# Optimization and calculus

**Tools > Optimization and Calculus** is one window that works with typed
formulas and programs. It minimizes a formula with the optimizers training
uses, solves linear and quadratic programs, and integrates and differentiates
a formula. Each answer is worked out in the background as you edit, and
**Stop** ends a long one. The **Portfolio Weights** node does the window's
portfolio work on every run of a graph. Python has the same work as
`pycyxwiz.calculus` and `pycyxwiz.optimize`.

The Tools > Optimization and Calculus menu items (Gradient Descent
Visualizer, Convexity Analyzer, Linear Programming (LP), Quadratic
Programming (QP), Numerical Differentiation, Numerical Integration) open this
window on that operation.

## The window

The operations are in three groups along the top.

| Group | Operation | What it answers |
| --- | --- | --- |
| Minimize | Optimizers | Where SGD, momentum, Adam, AdamW, NAdam, RMSprop and AdaGrad go on f(x, y) from a start, drawn on its contour map |
| Programs | Linear | The best c · x under linear constraints and bounds, any number of variables |
| Programs | Quadratic | The smallest ½ x′Q x + c · x under linear constraints; or a portfolio from a table of returns |
| Calculus | Integral | ∫ f(x) dx from a to b, adaptive with an error estimate, and the fixed rules beside it |
| Calculus | Derivative | f′(x) and f″(x) exactly from the formula, and by differences with their error against the step h |
| Calculus | Curvature | The gradient, the Hessian and its eigenvalues at a point of f(x, y), and a check over the range shown |

Under every answer, **Copy** copies it as tab-separated rows and **Export
CSV...** saves it. "Worked out in N ms" says how long it took.

### Formulas

A formula is typed as text, for example `(1 - x)^2 + 100*(y - x^2)^2`.

- Operators: `+ - * / %` and `^`. `^` is right-associative (`2^3^2` is
  2⁹), and binds tighter than a leading minus, so `-x^2` is −(x²).
- Numbers: `1.5`, `.5`, `2e-3`. Constants (any case): `pi`, `e`, `phi`.
- Functions (any case): `sin cos tan asin acos atan sinh cosh tanh sqrt cbrt
  exp log ln log10 log2 abs floor ceil round sign factorial`. With two
  arguments: `pow mod fmod min max atan2`.
- A name the operation does not take is refused with its place (`Unknown
  name 'z' at 5`). Unbalanced brackets, too, for example `Expected ')' at 12`.

Derivatives come exactly from the formula, as a formula. A formula that uses
`min`, `max`, `mod`, `fmod` or `factorial` has no exact derivative; the window
then shows the numerical one only. `floor`, `ceil`, `round` and `sign` count
as flat (derivative 0).

### Minimize: Optimizers

- **Formula** f(x, y), with **Examples**: Rosenbrock, Himmelblau, Rastrigin,
  Beale and Sphere.
- **Start** x and y, and **Steps** (up to 100,000).
- **Optimizers**: tick the ones to compare. Each has its learning rate
  (**lr**), and SGD and RMSprop also have **momentum**. They are the update
  rules training uses, in float32 as in training, and the gradient comes
  exactly from the formula.
- **Range shown**: the x and y range of the map.

The answer:

- a sentence naming the best result, and any optimizer that diverged, with
  its learning rate;
- the contour map with each path, the start and a diamond on a known minimum.
  When f spans many orders of magnitude, the lines are of log10(f − min + 1);
  the caption says so.
- **Step 10** shows the paths 10 steps further, and **Whole paths** shows
  them all;
- a table: optimizer, learning rate, where it ended, f, |grad f| there, and the
  result in words: Reached a minimum, Stopped on a saddle, Reached a flat
  point, Still moving at step N, Diverged at step N or Stopped at step N.

### Programs: Linear and Quadratic

- **Variables** (1 to 40) and **Maximize** / **Minimize** for a linear
  program.
- The objective row (c), one row per constraint (a name, the coefficients,
  ≤ ≥ or =, the right-hand side b), and each variable's **Lower bound** and
  **Upper bound**. Type `inf` or `-inf` for no limit. **+ Constraint** adds a
  row, and the bin at the start of a row removes it.
- **Paste rows**: one row a line, the coefficients, then `<=`, `>=` or `=`,
  then b, with an optional name ending in a colon (`Labour: 1 2 1 1 <= 40`).
  **Use these rows** replaces the rows.
- Quadratic adds **Q**, symmetric: typing an entry sets its mirror.
- **Examples**: Two products, Production plan, Diet, Transport, No solution;
  Quadratic (2 variables), Portfolio (3 assets), Box.

The answer:

- the status and a sentence with the best value and the point. Infeasible
  ("No point satisfies every constraint and bound together") and unbounded
  answers are given in words.
- a table of the variables;
- a table of the constraints: the room left, the **shadow price** (how much
  the best value changes when that row's b grows by 1) and whether it binds;
- with two variables, the feasible region shaded, each row's line and the
  best point. A quadratic program's contours are drawn on it too.

A Q that is not positive semidefinite is refused, naming its most negative
eigenvalue. Such a problem has no single minimum.

### Quadratic: From a returns table

**From a returns table** reads a portfolio problem from a table of returns,
one column per asset and one row per period:

- **Table of returns**: any table of the session (a loaded Data Input, a
  node's result);
- **Assets**: tick the asset columns. Untick a number column that is not a
  return, such as a period number.
- **Target return a period**: the least mean return the mix must earn; empty
  for the least risky mix;
- **Long only**: no weight below 0;
- **Read again**: reads the table again after a run changed it.

When a Data Split marked the table (a node result after the Split), the window
reads its Train rows, as the node does, and the note says how many other rows
it left out.

The window works out the means and the covariance (n − 1) of the rows on the
device, and solves: minimize ½ w′C w, with the weights summing to 1 (row
**Budget**) and earning at least the target (row **Return**). A row with a
missing value is left out, and the note under the answer says how many.

**Keep as a node: Portfolio weights** adds the same as a node reading that
table (a branch: the data path stays as it is). The message under the button
says what it added, and Ctrl+Z on the canvas undoes it.

### Calculus: Integral

- **Formula** f(x) and **From a to b**: numbers, `pi`, `e`, or `-inf` /
  `inf`.
- **Rule**: **Adaptive (Gauss–Kronrod)** splits where f is hard, and stops
  when its error estimate is below 1.5e-8 of the answer, as
  `scipy.integrate.quad`. **Trapezoid**, **Midpoint** and **Simpson** use
  equal intervals (Simpson an even count), and **Gauss–Legendre** uses n
  points (up to 1,000). The fixed rules take finite bounds.

The answer shows the value with its error estimate, the curve with the area
shaded, and a table of every rule's value, evaluations and difference from
the adaptive answer.

### Calculus: Derivative

- **Formula** f(x), **At** x, the **Differences** rule (Forward, Backward,
  Central) and **Step h** (a number above 0). Leave h empty for the step that
  suits float64 for that rule.

The answer shows f′ and f″ exactly (with their formulas) and numerically,
and a log-log plot of the numerical f′'s error against h. At large h the
rule's own error dominates: forward falls with h and central with h². At
small h rounding in f(x + h) − f(x) grows as h shrinks. The best h is where
the two meet.

### Calculus: Curvature

- **Formula** f(x, y), **Point** x and y, and **Range shown**.

The answer gives the gradient, the Hessian and its eigenvalues at this point,
and says what they mean there. All eigenvalues above 0 means it curves up in
every direction; mixed signs mean a saddle. "At this point" matters: one
point says nothing about the rest of the formula. The range check below it
counts the points of a 40 × 40 grid where the smallest eigenvalue is below 0
("Over the shown range (x -2..2, y -1..3, 40 × 40 points): not convex. The
Hessian has a negative eigenvalue at 672 of the 1,600 points, for example
(-0.05128, 0.02564)."), and the map marks the point. With no such point it
says "no point curves down, so f looks convex here (a check of samples, not a
proof)".

## The Portfolio Weights node

**Portfolio Weights** (Analytics) finds the least risky mix of assets from a
table of returns on every run of the graph:

- **asset_cols**: the asset columns (empty: every number column; the split
  marker is left out);
- **target_return**: the least mean return a period (empty: the
  minimum-variance mix);
- **long_only**: no weight below 0 (on by default).

It reads the Train rows when a Data Split marks them, otherwise every row,
and leaves out a row with a missing value. Its output is one row per asset:
`asset`, `weight`, `mean_return`, `sd` and `risk_share` (the asset's share
of the mix's variance, w_i (C w)_i / w′C w; the shares sum to 1). A target
no mix can earn is refused, naming the highest mean: "PortfolioWeights: no
mix of these assets earns 0.02 a period without short positions; the highest mean is 0.00935647
(growth)".

Example: `examples/cyxgraph/portfolio/` (five assets, 120 months).

## Python

`pycyxwiz.calculus` has `Expression`, `integrate`, `derivative`,
`step_errors`, `curvature`, `curvature_over_range` and `gauss_legendre`.
`pycyxwiz.optimize` has `linprog`, `qp`, `portfolio` and `follow`. They are
described with examples in `python_doc/pycyxwiz.md` (section 6.4). A worked
example checked against scipy and torch is
`examples/python/optimization_calculus.py`.

## Precision and checks

- Formulas are evaluated in float64: on the device when it has float64
  (CUDA, the CPU backend, most OpenCL GPUs), otherwise on the host.
- Programs are solved by HiGHS (the solver scipy's `linprog` uses) on the
  CPU. A quadratic program's objective is solved scaled to entries of size 1,
  so small covariances (1e-4 a month) are as exact as large ones.
- Optimizer paths are float32, as in training.
- Every number is checked by `test_computation_truth_optimization`
  (1,136 checks) against references made by
  `tests/computation_truth/reference/generate_optimization_fixtures.py`:
  - sympy: values and exact derivatives of 21 formulas;
  - scipy: `quad` and the fixed rules, 7 linear programs (values, shadow
    prices, room) and 4 quadratic programs;
  - numpy: Gauss-Legendre nodes, eigenvalues;
  - torch.optim: 9 optimizer paths, fed the same exact gradient;
  - an exact KKT solve: the five portfolio cases.

  The Portfolio Weights node is checked against the same KKT solve
  (`test_portfolio_weights`), and Python by `test_python_optimize.py`.

## Errors and limits

- The adaptive integral has no extrapolation (QUADPACK's QAGS has it), so an
  integrable singularity such as 1/√x on [0, 1] stops at 50 intervals with
  "not converged". Its error estimate (4e-8) still covers the error.
- The Typed and pasted programs take up to 40 variables. Python and the node
  take any number.
- Curvature over a range and the contour maps need a formula of x and y.
