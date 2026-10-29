"""Python subclasses of pycyxwiz.Module (TOFIX125 P3 step 3.2).

Softsign (no parameters) and Snake (one learnable parameter) are written only
in Python, added to a Sequential with Sequential.add, and trained by the C++
model. Run with the Python that pycyxwiz was built for:

    python test_python_modules.py <folder holding pycyxwiz> [<folder of its DLLs>]

Exit code 0 when every check passes.
"""
import gc
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pycyxwiz_test_env import Checks, import_pycyxwiz  # noqa: E402

cyx = import_pycyxwiz()
check = Checks()


def values(t):
    return [t.reshape([t.num_elements()]).at(i) for i in range(t.num_elements())]


class Softsign(cyx.Module):
    """x / (1 + |x|)"""

    def forward(self, x):
        self.x = x
        return x / (x.abs() + 1.0)

    def backward(self, grad_output):
        d = self.x.abs() + 1.0
        return grad_output / (d * d)


class Snake(cyx.Module):
    """x + sin^2(a x) / a with a learnable a"""

    def __init__(self, initial_a=1.0):
        super().__init__()
        self.add_parameter("a", cyx.Tensor.full([1], initial_a))

    def forward(self, x):
        self.x = x
        a = self.parameter("a").broadcast_to(x.shape())
        self.a = a
        self.s = (x * a).sin()
        return x + (self.s * self.s) / a

    def backward(self, grad_output):
        a = self.a
        sin2 = (self.x * a * 2.0).sin()
        da = (self.x * sin2) / a - (self.s * self.s) / (a * a)
        self.set_gradient("a", (grad_output * da).sum().reshape([1]))
        return grad_output * (sin2 + 1.0)


class Broken(cyx.Module):
    def forward(self, x):
        raise ValueError("deliberate failure in forward")

    def backward(self, grad_output):
        return grad_output


# --- the subclass is a module --------------------------------------------------
soft = Softsign()
x = cyx.Tensor.full([2, 3], 3.0)
y = soft.forward(x)
check(abs(values(y)[0] - 0.75) < 1e-6, "Softsign forward: 3 / (1 + 3) = 0.75")
check(soft.get_name() == "Softsign", "get_name is the Python class name")
check(not soft.has_parameters(), "Softsign has no parameters")

snake = Snake(0.5)
check(snake.has_parameters(), "Snake has a parameter")
check(snake.parameter_names() == ["a"], "parameter_names lists 'a'")
check(abs(snake.parameter("a").at(0) - 0.5) < 1e-6, "parameter('a') starts at 0.5")
check(set(snake.get_parameters().keys()) == {"a"}, "get_parameters returns {'a': ...}")

try:
    cyx.LinearModule(2, 2).add_parameter("w", cyx.Tensor.ones([1]))
    check(False, "add_parameter on a built-in module is refused")
except TypeError:
    check(True, "add_parameter on a built-in module is refused")

check(abs((1.0 / cyx.Tensor.full([1], 4.0)).at(0) - 0.25) < 1e-6, "scalar / Tensor")
mask = cyx.Tensor.full([2], 3.0) > 0.0  # a comparison gives a uint8 mask
check(values(1.0 - mask) == [0.0, 0.0], "scalar - uint8 mask is 1 - 1 = 0 (was 256: the mask was negated as uint8)")
first = values(cyx.Tensor.random([8], seed=7))
check(first == values(cyx.Tensor.random([8], seed=7)), "Tensor.random with a seed repeats its values")
check(first != values(cyx.Tensor.random([8], seed=8)), "another seed gives other values")
check(all(0.0 <= v < 1.0 for v in first), "seeded values are in [0, 1)")

# --- Sequential.add and training ---------------------------------------------
def make_data():
    # y = sum of sin(1.5 x_i): smooth and periodic, so Snake can help.
    xs = cyx.Tensor.random([64, 4]) * 4.0 - 2.0
    ys = (xs * 1.5).sin().sum(1, True)
    return xs, ys


def train(model, steps=150, lr=0.01):
    xs, ys = make_data()
    loss_fn = cyx.MSELoss()
    optimizer = cyx.Adam(lr)
    model.train()
    losses = []
    for _ in range(steps):
        prediction = model.forward(xs)
        losses.append(loss_fn.forward(prediction, ys).at(0))
        model.backward(loss_fn.backward(prediction, ys))
        model.update_parameters(optimizer)
    return losses


for name, make in (("Softsign", Softsign), ("Snake", lambda: Snake(1.0))):
    model = cyx.Sequential()
    model.add_linear(4, 16)
    activation = make()
    model.add(activation)
    model.add_linear(16, 1)
    check(len(model) == 3, name + ": Sequential.add appends the module")
    del activation
    gc.collect()  # the model keeps the Python object alive
    losses = train(model)
    check(losses[-1] < losses[0] * 0.5,
          "%s: the loss goes down (%.4f -> %.4f)" % (name, losses[0], losses[-1]))

snake = Snake(1.0)
model = cyx.Sequential()
model.add_linear(4, 16)
model.add(snake)
model.add_linear(16, 1)
train(model, steps=50)
a = snake.parameter("a").at(0)
check(abs(a - 1.0) > 1e-4, "Snake's a is trained by the optimizer (1.0 -> %.4f)" % a)
check("layer1.a" in model.get_parameters(), "the model lists Snake's a as layer1.a")

# --- a Python error names the module and keeps the traceback -----------------
model = cyx.Sequential()
model.add(Broken())
try:
    model.forward(cyx.Tensor.ones([2, 2]))
    check(False, "an error in forward stops the model")
except RuntimeError as error:
    text = str(error)
    check("Broken.forward raised" in text and "deliberate failure in forward" in text,
          "the error names the module and the Python message")
    check("test_python_modules.py" in text, "the error carries the Python traceback")

# --- a wrong return value -------------------------------------------------------
class ReturnsNumber(cyx.Module):
    def forward(self, x):
        return 1.0

    def backward(self, grad_output):
        return grad_output


try:
    ReturnsNumber().forward(cyx.Tensor.ones([1]))
    # Called directly from Python the method returns what it returns.
    model = cyx.Sequential()
    model.add(ReturnsNumber())
    model.forward(cyx.Tensor.ones([1]))
    check(False, "a forward that does not return a Tensor is refused")
except RuntimeError as error:
    check("must return a Tensor" in str(error), "a forward that does not return a Tensor is refused")

# --- gradcheck: the Engine's numerical gradient check (step 3.3) -------------
class WrongSoftsign(Softsign):
    """The classic slip: divides by (1 + |x|) instead of its square."""

    def backward(self, grad_output):
        return grad_output / (self.x.abs() + 1.0)


class WrongSnake(Snake):
    """Correct input gradient, parameter gradient off by a factor of 2."""

    def backward(self, grad_output):
        result = Snake.backward(self, grad_output)
        self.set_gradient("a", self.gradient("a") * 2.0)
        return result


for name, make, should_pass in (
        ("Softsign", lambda shape: Softsign(), True),
        ("Snake", lambda shape: Snake(1.0), True),
        ("WrongSoftsign", lambda shape: WrongSoftsign(), False),
        ("WrongSnake", lambda shape: WrongSnake(1.0), False)):
    result = cyx.gradcheck(make)
    check(bool(result) == should_pass and not result.error,
          "gradcheck %s: %s" % (name, result))
check("parameter a" == cyx.gradcheck(lambda shape: WrongSnake(1.0)).worst,
      "gradcheck says where the wrong gradient is (parameter a)")
check("input" == cyx.gradcheck(lambda shape: WrongSoftsign()).worst,
      "gradcheck says where the wrong gradient is (input)")

shapes_seen = []
cyx.gradcheck(lambda shape: shapes_seen.append(list(shape)) or Softsign(), input_shapes=[[5, 2]])
check(shapes_seen == [[5, 2]], "gradcheck passes each input shape, with the batch, to the factory")

broken = cyx.gradcheck(lambda shape: 42)
check(not broken and "must return a pycyxwiz.Module" in broken.error,
      "a factory that does not return a module is reported: %s" % broken)
raising = cyx.gradcheck(lambda shape: Broken())
check(not raising and "deliberate failure in forward" in raising.error,
      "an error in forward is reported, not raised: %s" % raising.error.splitlines()[0])

check.finish("test_python_modules")
