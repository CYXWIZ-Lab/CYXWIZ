"""Python subclasses of pycyxwiz.Loss (TOFIX125 P7 step 7.4).

Mean squared error and softmax cross entropy written only in Python give the
built-in losses' values and gradients; gradcheck_loss passes them and fails a
loss whose backward forgets the mean scaling. Run with the Python that
pycyxwiz was built for:

    python test_python_losses.py <folder holding pycyxwiz> [<folder of its DLLs>]

Exit code 0 when every check passes.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pycyxwiz_test_env import Checks, import_pycyxwiz  # noqa: E402

cyx = import_pycyxwiz()
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

check.finish("test_python_losses")
