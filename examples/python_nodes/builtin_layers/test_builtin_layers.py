# SPDX-License-Identifier: Apache-2.0
# Part of the CyxWiz plugin SDK examples (see cmake/sdk/NOTICE).
"""Checks the ten Python layers in builtin_layers.py.

    python test_builtin_layers.py <folder holding pycyxwiz> [<folder of its DLLs>]

1. Every node passes the gradient check with its default settings.
2. Each one computes what the Engine's built-in module computes: forward,
   the input gradient and, for Dense, the weight gradients (with the same
   weights). LayerNorm and Softplus have no Python-bound built-in, so they
   are checked against their formulas.
3. The bundle beside this file (nodes/builtin-layers.cyxnode) is current:
   its node.py is this folder's builtin_layers.py and every node is
   recorded as passing the gradient check.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# In the source tree the test helper is in cyxwiz-backend/python/tests.
for candidate in (os.path.join(HERE, "..", "..", "..", "cyxwiz-backend", "python", "tests"), HERE):
    if os.path.isfile(os.path.join(candidate, "pycyxwiz_test_env.py")):
        sys.path.insert(0, candidate)
        break
sys.path.insert(0, HERE)
from pycyxwiz_test_env import Checks, import_pycyxwiz  # noqa: E402

cyx = import_pycyxwiz()
from pycyxwiz import nodes  # noqa: E402

import builtin_layers as b  # noqa: E402

check = Checks()
ALL = [b.Dense, b.LayerNorm, b.ReLU, b.LeakyReLU, b.ELU, b.Sigmoid, b.Tanh, b.Softplus, b.GELU, b.SiLU]
TOLERANCE = 1e-5


def values(t):
    return [t.reshape([t.num_elements()]).at(i) for i in range(t.num_elements())]


def largest_difference(a, b_):
    va, vb = values(a), values(b_)
    if len(va) != len(vb):
        return float("inf")
    return max((abs(x - y) for x, y in zip(va, vb)), default=0.0)


def uniform(shape, low, high):
    return cyx.Tensor.random(shape) * (high - low) + low


print("1. the gradient check, default settings")
check(len(ALL) == 10, "ten nodes")
for cls in ALL:
    result = nodes.gradcheck(cls, {})
    check(result.passed, "%s: %s" % (cls.__name__, result.error or "largest relative error %.1e" % result.max_relative_error))

print("2. the same numbers as the built-in modules")
x = uniform([4, 6], -3.0, 3.0)
g = uniform([4, 6], -1.0, 1.0)
pairs = [
    (b.ReLU, {}, cyx.ReLUModule()),
    (b.LeakyReLU, {"negative_slope": "0.01"}, cyx.LeakyReLUModule(0.01)),
    (b.ELU, {"alpha": "1.0"}, cyx.ELUModule(1.0)),
    (b.Sigmoid, {}, cyx.SigmoidModule()),
    (b.Tanh, {}, cyx.TanhModule()),
    (b.GELU, {}, cyx.GELUModule()),
    (b.SiLU, {}, cyx.SwishModule()),
]
for cls, settings, builtin in pairs:
    ours = nodes.create(cls, settings, [6])
    forward = largest_difference(ours.forward(x), builtin.forward(x))
    backward = largest_difference(ours.backward(g), builtin.backward(g))
    check(forward < TOLERANCE and backward < TOLERANCE,
          "%s = %s (forward %.1e, gradient %.1e)" % (cls.__name__, type(builtin).__name__, forward, backward))

# Dense against LinearModule with the same weights (LinearModule keeps W as
# [out, in], the Python node as [in, out]).
dense = nodes.create(b.Dense, {"units": "3"}, [6])
linear = cyx.LinearModule(6, 3, True)
linear.set_parameters({"weight": dense.parameter("weight").transpose(), "bias": dense.parameter("bias")})
forward = largest_difference(dense.forward(x), linear.forward(x))
g3 = uniform([4, 3], -1.0, 1.0)
backward = largest_difference(dense.backward(g3), linear.backward(g3))
linear_gradients = linear.get_gradients()
weight_gradient = largest_difference(dense.gradient("weight").transpose(), linear_gradients["weight"])
bias_gradient = largest_difference(dense.gradient("bias"), linear_gradients["bias"])
check(max(forward, backward, weight_gradient, bias_gradient) < TOLERANCE,
      "Dense = LinearModule (forward %.1e, input gradient %.1e, weight gradient %.1e, bias gradient %.1e)"
      % (forward, backward, weight_gradient, bias_gradient))
check(nodes.output_shape(b.Dense, {"units": "5"}, [7, 6]) == [7, 5], "Dense on a sequence [7, 6] -> [7, 5]")

# LayerNorm: each row to mean 0, variance 1 (gain 1, shift 0 at the start).
norm = nodes.create(b.LayerNorm, {}, [6])
y = values(norm.forward(x))
worst = 0.0
for row in range(4):
    out = y[row * 6:(row + 1) * 6]
    mean = sum(out) / 6.0
    variance = sum((v - mean) ** 2 for v in out) / 6.0
    worst = max(worst, abs(mean), abs(variance - 1.0))
check(worst < 1e-3, "LayerNorm rows have mean 0 and variance 1 (largest error %.1e)" % worst)

# Softplus: log(1 + e^x), and its derivative is the sigmoid.
soft = nodes.create(b.Softplus, {}, [6])
xs = values(x)
expected = [math.log1p(math.exp(v)) for v in xs]
worst = max(abs(a - e) for a, e in zip(values(soft.forward(x)), expected))
gs = values(g)
expected_grad = [gv / (1.0 + math.exp(-v)) for gv, v in zip(gs, xs)]
worst_grad = max(abs(a - e) for a, e in zip(values(soft.backward(g)), expected_grad))
check(worst < TOLERANCE and worst_grad < TOLERANCE,
      "Softplus = log(1 + e^x) (forward %.1e, gradient %.1e)" % (worst, worst_grad))
big = cyx.Tensor.full([2], 100.0)
check(abs(values(soft.forward(big))[0] - 100.0) < 1e-3, "Softplus(100) = 100, no overflow")

print("3. the bundle beside this file is current")
bundle = os.path.join(HERE, "nodes", "builtin-layers.cyxnode")
with open(os.path.join(HERE, "builtin_layers.py"), "rb") as source, \
        open(os.path.join(bundle, "node.py"), "rb") as bundled:
    # Line endings aside: git may rewrite the .py on checkout; node.py is byte-exact.
    check(source.read().replace(b"\r\n", b"\n") == bundled.read().replace(b"\r\n", b"\n"),
          "nodes/builtin-layers.cyxnode/node.py is builtin_layers.py (bundle again after editing it)")
with open(os.path.join(bundle, "node.json"), encoding="utf-8") as handle:
    manifest = json.load(handle)
names = sorted(n["name"] for n in manifest["nodes"])
check(names == sorted(cls.__name__ for cls in ALL), "node.json lists the ten nodes")
not_passed = [n["name"] for n in manifest["nodes"] if n["verification"]["status"] != "passed"]
check(not not_passed, "every node is recorded as passing the gradient check" + (": not " + ", ".join(not_passed) if not_passed else ""))

check.finish("test_builtin_layers")
