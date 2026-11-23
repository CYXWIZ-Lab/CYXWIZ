# Made with Make a Python node in CyxWiz.
# The decorator and the settings follow the form; the method bodies are yours.
import pycyxwiz as cyx
from pycyxwiz import nodes


@nodes.loss(
    id="project.catdog.log_cosh",
    name="LogCosh",
    version="1.0.0",
    target="continuous",
    category="Training",
    display_name="Log-cosh",
    description="mean(log(cosh(prediction - target))): like MSE near 0, like L1 far away",
)
class LogCosh(cyx.Loss):
    def forward(self, predictions, targets):
        d = predictions - targets
        cosh = (d.exp() + (-d).exp()) * 0.5
        return cosh.log().mean()

    def backward(self, predictions, targets):
        e2 = ((predictions - targets) * 2.0).exp()
        tanh = (e2 - 1.0) / (e2 + 1.0)
        return tanh * (1.0 / predictions.num_elements())
