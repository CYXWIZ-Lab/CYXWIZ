# Made with Make a Python node in CyxWiz.
# The decorator and the settings follow the form; the method bodies are yours.
import pycyxwiz as cyx
from pycyxwiz import nodes


@nodes.activation(
    id="project.catdog.soft_clip",
    name="SoftClip",
    version="1.0.0",
    category="Activation",
    display_name="Soft Clip",
    description="x / (1 + |x|): like tanh, cheaper, never saturates to exactly ±1",
)
class SoftClip:
    def f(self, x):
        return x / (x.abs() + 1.0)

    def df(self, x):
        d = x.abs() + 1.0
        return 1.0 / (d * d)
