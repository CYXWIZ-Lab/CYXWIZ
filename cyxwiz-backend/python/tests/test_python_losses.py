"""Python subclasses of pycyxwiz.Loss (TOFIX125 P7 steps 7.4 and 7.5).

Mean squared error and softmax cross entropy written only in Python give the
built-in losses' values and gradients; gradcheck_loss passes them and fails a
loss whose backward forgets the mean scaling. @nodes.loss describes a loss as a
node: its pins, its target kind, settings, the check and the bundle entry.
Run with the Python that
pycyxwiz was built for:

    python test_python_losses.py <folder holding pycyxwiz> [<folder of its DLLs>]

Exit code 0 when every check passes.
"""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pycyxwiz_test_env import Checks, import_pycyxwiz  # noqa: E402

cyx = import_pycyxwiz()
from pycyxwiz import nodes  # noqa: E402
check = Checks()


def values(t):
    return [t.reshape([t.num_elements()]).at(i) for i in range(t.num_elements())]


def largest_difference(a, b):
    va, vb = values(a), values(b)
    if len(va) != len(vb):
        return float("inf")
    return max((abs(x - y) for x, y in zip(va, vb)), default=0.0)


class MeanSquaredError(cyx.Loss):
    """mean((p - t)^2)"""

    def forward(self, predictions, targets):
        d = predictions - targets
        return (d * d).mean()

    def backward(self, predictions, targets):
        return (predictions - targets) * (2.0 / predictions.num_elements())


class SoftmaxCrossEntropy(cyx.Loss):
    """mean over rows of -sum(t * log softmax(p)); targets one-hot"""

    @staticmethod
    def softmax(p):
        last = p.num_dimensions() - 1
        e = (p - p.max(last, True)).exp()
        return e / e.sum(last, True)

    def forward(self, predictions, targets):
        rows = predictions.num_elements() // predictions.shape()[-1]
        return (targets * self.softmax(predictions).log()).sum() * (-1.0 / rows)

    def backward(self, predictions, targets):
        rows = predictions.num_elements() // predictions.shape()[-1]
        return (self.softmax(predictions) - targets) * (1.0 / rows)


@nodes.loss("com.example.losses", "Huber", "1.0.0", target="continuous")
class Huber(cyx.Loss):
    """Squared below delta, linear above"""
    delta = nodes.Param(float, default=1.0, min=0.01, description="Where the loss turns linear")

    def forward(self, predictions, targets):
        d = (predictions - targets).abs()
        small = cyx.Tensor.ones(d.shape()) * (d <= self.delta)
        values = small * (d * d * 0.5) + (1.0 - small) * (d * self.delta - 0.5 * self.delta * self.delta)
        return values.mean()

    def backward(self, predictions, targets):
        d = predictions - targets
        return d.clip(-self.delta, self.delta) * (1.0 / predictions.num_elements())


@nodes.loss("com.example.losses", "SoftmaxCE", "1.0.0")
class SoftmaxCE(SoftmaxCrossEntropy):
    """Softmax cross entropy over one-hot targets"""


class ForgetsTheMean(MeanSquaredError):
    def backward(self, predictions, targets):
        return (predictions - targets) * 2.0


class Broken(cyx.Loss):
    def forward(self, predictions, targets):
        raise ValueError("no loss today")

    def backward(self, predictions, targets):
        return predictions


print("1. a Python loss is a pycyxwiz.Loss")
mse = MeanSquaredError()
check(isinstance(mse, cyx.Loss), "MeanSquaredError is a pycyxwiz.Loss")
check(isinstance(cyx._loss_address(mse), int) and cyx._loss_address(mse) != 0, "_loss_address gives its address")
check(mse.get_name() == "MeanSquaredError", "its name is the class name: " + mse.get_name())

print("2. the same numbers as the built-in losses")
p = cyx.Tensor.random([4, 3]) * 4.0 - 2.0
t = cyx.Tensor.random([4, 3]) * 2.0 - 1.0
builtin = cyx.MSELoss()
forward = largest_difference(mse.forward(p, t), builtin.forward(p, t))
backward = largest_difference(mse.backward(p, t), builtin.backward(p, t))
check(forward < 1e-5 and backward < 1e-5, "MSE = MSELoss (forward %.1e, gradient %.1e)" % (forward, backward))

labels = cyx.Tensor.zeros([4, 3])
for row, c in enumerate([0, 2, 1, 2]):
    labels.set(row, c, 1.0)
ce = SoftmaxCrossEntropy()
builtin_ce = cyx.CrossEntropyLoss()
forward = largest_difference(ce.forward(p, labels), builtin_ce.forward(p, labels))
backward = largest_difference(ce.backward(p, labels), builtin_ce.backward(p, labels))
check(forward < 1e-5 and backward < 1e-5,
      "softmax cross entropy = CrossEntropyLoss (forward %.1e, gradient %.1e)" % (forward, backward))

print("3. gradcheck_loss")
result = cyx.gradcheck_loss(MeanSquaredError, targets="values")
check(result.passed, "MSE passes with value targets: %s" % result)
result = cyx.gradcheck_loss(SoftmaxCrossEntropy, targets="one_hot")
check(result.passed, "softmax cross entropy passes with one-hot targets: %s" % result)
result = cyx.gradcheck_loss(ForgetsTheMean, targets="values")
check(not result.passed and not result.error, "a backward without the mean scaling fails: %s" % result)
result = cyx.gradcheck_loss(Broken)
check(not result.passed and "no loss today" in result.error and "Broken.forward" in result.error,
      "a Python error is reported with the class and method: %s" % result.error)
try:
    cyx.gradcheck_loss(MeanSquaredError, targets="weights")
    check(False, "unknown target kinds are refused")
except ValueError as e:
    check("one_hot" in str(e), "unknown target kinds are refused: %s" % e)

print("4. @nodes.loss")
check(nodes.is_loss(Huber) and nodes.is_node(Huber) and not nodes.is_loss(MeanSquaredError), "is_loss")
info = nodes.describe(Huber)
check(info["role"] == "loss" and info["tier"] == "loss" and info["target"] == "continuous"
      and info["category"] == "Loss", "describe: role, tier, target, category")
check(info["inputs"] == [{"name": "Predictions", "type": "Tensor"}, {"name": "Targets", "type": "Labels"}]
      and info["outputs"] == [{"name": "Loss", "type": "Loss"}] and "shape_rule" not in info,
      "describe: the built-in losses' pins and no shape rule")
check([q["name"] for q in info["parameters"]] == ["delta"], "describe: its setting")
check(nodes.describe(SoftmaxCE)["target"] == "categorical", "the default target is categorical")

huber = nodes.create(Huber, {"delta": "0.5"})
check(isinstance(huber, cyx.Loss) and abs(huber.delta - 0.5) < 1e-9, "create converts the setting")
try:
    nodes.create(Huber, {"delta": "0"})
    check(False, "a setting below its minimum is refused")
except ValueError as e:
    check("at least" in str(e), "a setting below its minimum is refused: %s" % e)
for cls, kind in ((Huber, "values"), (SoftmaxCE, "one-hot")):
    result = nodes.gradcheck(cls)
    check(result.passed, "%s passes the check with %s targets: %s" % (cls.__name__, kind, result))
try:
    nodes.output_shape(Huber, {}, [3])
    check(False, "a loss has no output shape")
except TypeError as e:
    check("is a loss" in str(e), "a loss has no output shape: %s" % e)

for bad, expected in (
        (lambda: nodes.loss("com.example.losses", "X", "1.0.0")(type("OnlyForward", (cyx.Loss,), {
            "forward": lambda s, p, t: p})), "needs backward"),
        (lambda: nodes.loss("com.example.losses", "X", "1.0.0")(type("NotALoss", (cyx.Module,), {})),
         "subclass of pycyxwiz.Loss"),
        (lambda: nodes.loss("com.example.losses", "X", "1.0.0", target="ranked")(MeanSquaredError), "target"),
        (lambda: nodes.node("com.example.losses", "X", "1.0.0", role="loss")(type("M", (cyx.Module,), {
            "forward": lambda s, x: x, "backward": lambda s, g: g})), "role")):
    try:
        bad()
        check(False, "refused: " + expected)
    except (TypeError, ValueError) as e:
        check(expected in str(e), "refused (%s): %s" % (expected, e))

folder = tempfile.mkdtemp(prefix="cyxwiz_losses_")
try:
    path = nodes.bundle(Huber, SoftmaxCE, folder=folder, name="my-losses")
    with open(os.path.join(path, "node.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = {n["name"]: n for n in manifest["nodes"]}
    check(set(entries) == {"Huber", "SoftmaxCE"}, "the bundle lists both losses")
    check(entries["Huber"]["role"] == "loss" and entries["Huber"]["target"] == "continuous"
          and "shape_rule" not in entries["Huber"] and len(entries["Huber"]["inputs"]) == 2,
          "a loss entry: role, target, two inputs, no shape rule")
    check(all(n["verification"]["status"] == "passed" for n in manifest["nodes"]),
          "both are recorded as passing the check")
finally:
    shutil.rmtree(folder, ignore_errors=True)

check.finish("test_python_losses")
