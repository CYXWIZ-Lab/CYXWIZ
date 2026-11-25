---
title: pycyxwiz examples
description: The runnable pycyxwiz scripts in examples/python, what each one shows, what it needs and how to run it, with the output of one run.
type: reference
area: python
audience: user
status: experimental
since: 1.0.8
owner: python
last_reviewed: 2026-11-25
related:
  - quickstart.md
  - engine-and-standalone.md
  - evaluation-and-metrics.md
  - maths-and-optimization.md
  - signals-statistics-and-text.md
---

The scripts in `examples/python` of the source repository use pycyxwiz end
to end. None needs a download: each makes its data with a fixed seed or
reads a file committed next to it, and most check their answers against
NumPy, SciPy, scikit-learn or PyTorch.

> Checked against pycyxwiz 0.2.0. Every script below was run with it and
> printed what is shown.

## How to run them

From the repository root, with the Python 3.12 pycyxwiz is built for, name
the folder that holds pycyxwiz and, if different, the folder of its DLLs:

<!-- not run: a shell command; the folders depend on your build -->
```bash
python examples/python/pycyxwiz_basic.py build/windows-release/lib/Release build/windows-release/bin/Release
```

With an installed Engine, both folders are the Engine's base folder
([Run in the Engine or standalone](engine-and-standalone.md#outside-the-engine)).
`model_tools.py`, `system_memory.py` and `utilities.py` import the helper
`cyxwiz-backend/python/tests/pycyxwiz_test_env.py`, so they run from the
source repository.

## The scripts

| Script | What it shows | Needs |
| --- | --- | --- |
| `pycyxwiz_basic.py` | The backend, the devices, tensors, factories, arithmetic, NumPy and the data types | numpy |
| `ex1_pycyxwiz.py` | Linear regression by gradient descent with `pycyxwiz.linalg` on `ex1data1.txt` (profit against city population), with plots of the fit and the cost surface | numpy, matplotlib |
| `model_tools.py` | `pycyxwiz.model`: the summary, a learning rate range test, training at the suggested rate, saliency and integrated gradients for a wrong row, and temperature calibration | numpy |
| `linear_algebra_calibration.py` | Least squares, rank, SVD, QR, Cholesky and eigenvalues on made-up sensor readings, each checked against `numpy.linalg` | numpy |
| `optimization_calculus.py` | A production plan with shadow prices, a portfolio from `examples/cyxgraph/portfolio/returns.csv`, integrals, derivatives and optimizer paths, checked against SciPy | numpy, scipy |
| `text_words_and_sentiment.py` | `pycyxwiz.text` on short reviews: words, stop words, word pairs, stems and VADER scores | nothing else |
| `utilities.py` | `pycyxwiz.utilities`: a formula, units, hashes, a JSON mistake found, regex, seeded random values, UUIDs | nothing else |
| `system_memory.py` | `pycyxwiz.system`: device memory before and after a tensor, releasing it, the device's free memory, host tensors | nothing else |

The other scripts in the folder do not use pycyxwiz:
`cats_dogs_embedded_inference.py` and `codefeedback_text_inference.py` send
requests to a deployed model's inference server, and
`prepare_codefeedback_demo_dataset.py` and
`prepare_sentiment_analysis_demo.py` prepare data and vocabularies for
example graphs.

## What they print

`model_tools.py`:

```text
Linear(4 -> 32)  [32]    160 parameters   128 multiply-adds
ReLU             [32]      0 parameters     0 multiply-adds
Linear(32 -> 3)  [3]      99 parameters    96 multiply-adds
total 259 parameters, 224 multiply-adds a sample
range test: 97 steps; lowest loss 0.626 at 0.534; steepest fall at 0.0187
test accuracy 0.830 on 200 rows
row 2: a 0 read as 2; saliency [1.547, 1.267, 2.063, 1.099] IG [-1.943, 1.879, 2.272, -0.344] (sum 1.864, score moved 1.872)
calibration error 0.0724; temperature 1.283 brings it to 0.0562
```

`linear_algebra_calibration.py`:

```text
weights: s1 2.0006, s2 -0.4993, s3 1.4998, offset 0.3000 (made with 2, -0.5, 1.5, 0.3)
A: 200 x 4, rank 4, condition 1.4
with s4 = s1 + s2: rank 4 of 5; shortest weights s1 1.5001, s2 -0.9997, s3 1.4998, s4 0.5004, offset 0.3000
singular values of the centred readings: 14.06, 13.49, 11.19
PCA from the covariance: variances 0.9937, 0.9150, 0.6291 (= singular values squared / 199)
QR gives A back: 8.9e-16; Cholesky of the covariance gives it back: 1.1e-16
every answer matches numpy.linalg
```

`optimization_calculus.py`:

```text
plan: 10 chairs, 40 tables, 5 desks earn 2500.00
  carpentry has 30.00 hours to spare; one more hour of finishing is worth 20.00, of packing 10.00; each desk ordered beyond 5 costs 10.00
least risky mix: equity 0.0575, growth 0.0060, credit 0.0000, bonds 0.9365, gold 0.0000 (sd 1.0912 % a month)
earning 0.50 % a month: equity 0.0847, growth 0.3125, credit 0.2029, bonds 0.4000, gold 0.0000 (sd 2.6530 % a month)
P(-1 < Z < 1) = 0.682689492137086 (21 evaluations), over the whole line 1.000000000000000
f'(1) of x^3 * exp(-x): exact 0.7357588823, central differences 0.7357588823 (best h about 1.8e-06)
Rosenbrock from (-2, 2), 2000 steps: adam ends at (0.6455, 0.4158), sgd + momentum at (0.9999, 0.9998)
every answer matches scipy
```

`utilities.py` (the end; the UUIDs differ every run):

```text
== Random (seed 42: the same values on every platform)
Normal(10, 2), 1,000 values: mean 9.9239, SD (n - 1) 2.0449, median 9.9598, Q1 8.5596, Q3 11.2511
first values: [7.8457, 12.189, 13.5895, 8.1757, 8.7654]
6,000 dice throws, counts of 1..6: [1045, 986, 975, 996, 989, 1009]
```

`system_memory.py`, on a machine with an NVIDIA GTX 1050 Ti (the figures
depend on the machine):

```text
before initialize(): False
after initialize()           allocated        0 B   in use        0 B   buffers 0 (0 in use)
with a 16 MB tensor          allocated   16.00 MB   in use   16.00 MB   buffers 1 (1 in use)
tensor dropped               allocated   16.00 MB   in use        0 B   buffers 1 (0 in use)
released                     16.00 MB (16.00 MB -> 0 B)
after the release            allocated        0 B   in use        0 B   buffers 0 (0 in use)
device                       3.77 GB free of 4.00 GB (NVIDIA management library)
host tensors                 0 B now, peak 0 B
peak after reset             0 B
```

`text_words_and_sentiment.py` prints the counts, pairs, stems and scores
listed at the top of the script; `pycyxwiz_basic.py` and `ex1_pycyxwiz.py`
print each step as they go (`ex1_pycyxwiz.py` ends with the predicted profit
for cities of 35,000 and 70,000: $4,519.77 and $45,342.45).

## More examples

- [Make your first Python node](../sdk/get-started/first-python-node.md)
  and `examples/python_nodes`: Python modules as nodes.
- `cyxwiz-engine/python/examples`: matplotlib figures and RL metrics shown
  in the Engine.
