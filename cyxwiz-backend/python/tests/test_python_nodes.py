"""pycyxwiz.nodes (TOFIX125 P3 step 3.4): Softsign as a tier 1 activation
(f and df only) and Snake as a tier 2 node with a setting, both written only
in Python. Run like test_python_modules.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pycyxwiz_test_env import Checks, import_pycyxwiz  # noqa: E402

cyx = import_pycyxwiz()
check = Checks()

from pycyxwiz import nodes  # noqa: E402
import pycyxwiz.nodes  # noqa: E402,F401

check(pycyxwiz.nodes is nodes and cyx.nodes is nodes, "pycyxwiz.nodes imports both ways")


from example_nodes import Snake, Softsign  # noqa: E402


@nodes.node(id="com.example.layers", name="Halve", version="2.1.0")
class Halve(cyx.Module):
    """Keeps the first half of the features."""

    def forward(self, x):
        return x

    def backward(self, grad_output):
        return grad_output

    def output_shape(self, input_shape):
        return input_shape[:-1] + [input_shape[-1] // 2]


# --- tier 1 ---------------------------------------------------------------------
check(issubclass(Softsign, cyx.Module), "@activation makes a pycyxwiz.Module")
soft = nodes.create(Softsign)
y = soft.forward(cyx.Tensor.full([2, 3], 3.0))
check(abs(y.at(0, 0) - 0.75) < 1e-6, "tier 1 forward is f(x)")
g = soft.backward(cyx.Tensor.ones([2, 3]))
check(abs(g.at(0, 0) - 1.0 / 16.0) < 1e-6, "tier 1 backward is grad * df(x)")
result = nodes.gradcheck(Softsign)
check(bool(result), "tier 1 Softsign passes the gradient check: %s" % result)

# --- tier 2 with a setting ------------------------------------------------------
snake = nodes.create(Snake, {"initial_a": "2.5"}, [16])
check(snake.initial_a == 2.5, "a text setting is converted to float")
check(abs(snake.parameter("a").at(0) - 2.5) < 1e-6, "build() made a from the setting")
check(nodes.create(Snake, {}, [16]).initial_a == 1.0, "a missing setting takes its default")
for settings, expected in (({"initial_a": "-1"}, "initial_a must be at least 0.01"),
                           ({"initial_a": "abc"}, "initial_a must be a number, not 'abc'"),
                           ({"speed": "1"}, "Snake has no setting speed")):
    try:
        nodes.create(Snake, settings, [16])
        check(False, "refused: %s" % expected)
    except ValueError as error:
        check(str(error) == expected, "refused: %s" % error)
try:
    snake.initial_a = 3.0
    check(False, "a setting is read-only inside the node")
except AttributeError:
    check(True, "a setting is read-only inside the node")
check(bool(nodes.gradcheck(Snake, {"initial_a": "1.0"})), "tier 2 Snake passes the gradient check")

# --- shapes -------------------------------------------------------------------------
check(nodes.output_shape(Snake, {}, [7, 16]) == [7, 16], "same_as_input when output_shape is not defined")
check(nodes.output_shape(Halve, {}, [7, 16]) == [7, 8], "output_shape is called when defined")

# --- describe: the manifest entry -------------------------------------------------
info = nodes.describe(Snake)
check(info["id"] == "com.example.acts" and info["name"] == "Snake" and info["version"] == "1.0.0",
      "describe: identity")
check(info["tier"] == "module" and info["role"] == "activation" and info["category"] == "Activation",
      "describe: tier, role, category")
check(info["shape_rule"] == "same_as_input", "describe: shape rule")
check(info["parameters"] == [{"name": "initial_a", "type": "float", "default": "1.0",
                              "validation": "0.01-", "description": "Starting value of a",
                              "choices": []}],
      "describe: parameters as ParameterDefinition fields")
check(info["inputs"] == [{"name": "Input", "type": "Tensor"}], "describe: one Tensor input")
check(nodes.describe(Softsign)["tier"] == "activation", "describe: tier 1 is 'activation'")
check(nodes.describe(Halve)["description"] == "Keeps the first half of the features.",
      "describe: the docstring's first line when no description is given")
check(nodes.describe(Halve)["shape_rule"] == "python", "describe: shape rule python")

# --- wrong declarations are refused at decoration ------------------------------------
for kwargs, text in (({"id": "Com.Example", "name": "X", "version": "1.0.0"}, "must be lower case"),
                     ({"id": "com.example", "name": "1x", "version": "1.0.0"}, "must start with a letter"),
                     ({"id": "com.example", "name": "X", "version": "1.0"}, "must look like 1.0.0")):
    try:
        nodes.node(**kwargs)(type("X", (cyx.Module,), {"forward": lambda s, x: x,
                                                        "backward": lambda s, g: g}))
        check(False, "refused: " + text)
    except ValueError as error:
        check(text in str(error), "refused: %s" % error)
try:
    nodes.activation(id="com.example", name="NoDerivative", version="1.0.0")(
        type("NoDerivative", (), {"f": lambda s, x: x}))
    check(False, "an activation without df is refused")
except TypeError as error:
    check("df(self, x)" in str(error), "an activation without df is refused")

# --- both train through Sequential.add -----------------------------------------------
xs = cyx.Tensor.random([64, 4]) * 4.0 - 2.0
ys = (xs * 1.5).sin().sum(1, True)
for name, module in (("Softsign", nodes.create(Softsign)),
                     ("Snake", nodes.create(Snake, {"initial_a": "1.0"}, [16]))):
    model = cyx.Sequential()
    model.add_linear(4, 16)
    model.add(module)
    model.add_linear(16, 1)
    loss_fn, optimizer = cyx.MSELoss(), cyx.Adam(0.01)
    model.train()
    first = last = None
    for _ in range(150):
        prediction = model.forward(xs)
        last = loss_fn.forward(prediction, ys).at(0)
        first = last if first is None else first
        model.backward(loss_fn.backward(prediction, ys))
        model.update_parameters(optimizer)
    check(last < first * 0.5, "%s node trains (%.4f -> %.4f)" % (name, first, last))

check.finish("test_python_nodes")
