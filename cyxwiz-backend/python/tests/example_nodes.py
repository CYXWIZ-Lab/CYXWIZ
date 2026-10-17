"""Example nodes written only in Python (TOFIX125 P3): Softsign as a tier 1
activation and Snake as a tier 2 node with a setting and a learnable
parameter. Imported by test_python_nodes.py and bundled by
test_python_bundle.py; docs/extending/python-nodes.md shows the same code.
"""
import pycyxwiz as cyx
from pycyxwiz import nodes


@nodes.activation(id="com.example.acts", name="Softsign", version="1.0.0",
                  description="x / (1 + |x|)")
class Softsign:
    def f(self, x):
        return x / (x.abs() + 1.0)

    def df(self, x):
        d = x.abs() + 1.0
        return 1.0 / (d * d)


@nodes.node(id="com.example.acts", name="Snake", version="1.0.0", category="Activation",
            role="activation", description="x + sin^2(a x) / a, learnable a")
class Snake(cyx.Module):
    initial_a = nodes.Param(float, default=1.0, min=0.01, description="Starting value of a")

    def build(self, input_shape):
        self.add_parameter("a", cyx.Tensor.full([1], self.initial_a))

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


@nodes.activation(id="com.example.acts", name="SoftsignWrong", version="1.0.0",
                  description="Softsign with the classic slip in df, for the tests")
class SoftsignWrong:
    def f(self, x):
        return x / (x.abs() + 1.0)

    def df(self, x):
        return 1.0 / (x.abs() + 1.0)
