# Made with Make a Python node in CyxWiz.
# The decorator and the settings follow the form; the method bodies are yours.
import pycyxwiz as cyx
from pycyxwiz import nodes


@nodes.node(
    id="project.catdog.swish",
    name="Swish",
    version="1.0.0",
    role="activation",
    category="Activation",
    display_name="Swish (beta)",
    description="x * sigmoid(beta * x); beta = 1 is SiLU",
)
class Swish(cyx.Module):
    beta = nodes.Param(float, default=1.0, min=0.0, description="How sharp the curve is")

    def forward(self, x):
        self.x = x
        self.s = 1.0 / ((-(x * self.beta)).exp() + 1.0)
        return x * self.s

    def backward(self, grad_output):
        s = self.s
        return grad_output * (s + self.x * self.beta * s * (1.0 - s))
