# SPDX-License-Identifier: Apache-2.0
# Part of the CyxWiz plugin SDK examples (see cmake/sdk/NOTICE).
"""Ten CyxWiz layers and four losses written again as Python nodes.

Each node does what the built-in node of the same name does, written only
with pycyxwiz tensor operations, so you can read how a layer or a loss is
made and copy it. Put the layers between a Data Input and a loss like the
built-in ones, and a loss after the last layer with the Data Input's labels
on its Targets pin; the example graph blobs_python_layers.cyxgraph does both.

    Dense      y = x W + b, learnable W [in, units] and b [units]
    LayerNorm  normalise the last dimension, learnable gain and shift
    ReLU, LeakyReLU, ELU, Sigmoid, Tanh, Softplus, GELU (tanh form), SiLU
    CrossEntropy  softmax cross entropy over one-hot targets (classification)
    MSE, L1, Huber  mean squared, absolute and Huber error (regression)

Bundle them (writes nodes/builtin-layers.cyxnode/):

    from pycyxwiz import nodes
    import builtin_layers as b
    nodes.bundle(*b.ALL, folder="nodes", name="builtin-layers")
"""
import math

import pycyxwiz as cyx
from pycyxwiz import nodes

ID = "com.cyxwiz.examples.layers"
CATEGORY = "Python examples"
GELU_C = math.sqrt(2.0 / math.pi)


# ---------------------------------------------------------------- helpers

def matmul(a, b):
    """[m, k] x [k, n] -> [m, n] on the device (batch_matmul over one batch)."""
    return a.unsqueeze(0).batch_matmul(b.unsqueeze(0)).squeeze(0)


def rows(x):
    """x as [rows, last dimension]: every leading dimension folded into rows."""
    shape = x.shape()
    return x.reshape([x.num_elements() // shape[-1], shape[-1]])


def sigmoid(x):
    return 1.0 / ((-x).exp() + 1.0)


def tanh(x):
    return 2.0 / ((x * -2.0).exp() + 1.0) - 1.0


def relu_mask(x):
    """1.0 where x > 0, else 0.0, as float32 (a comparison alone gives uint8)."""
    return cyx.Tensor.ones(x.shape()) * (x > 0.0)


def softmax(x):
    """Softmax over the last dimension, shifted by the row maximum (no overflow)."""
    last = x.num_dimensions() - 1
    e = (x - x.max(last, True)).exp()
    return e / e.sum(last, True)


def row_count(x):
    return x.num_elements() // x.shape()[-1]


# ----------------------------------------------------------------- layers

@nodes.node(id=ID, name="Dense", version="1.0.0", category=CATEGORY, role="layer",
            display_name="Dense (Python)",
            description="y = x W + b over the last dimension, like the built-in Dense")
class Dense(cyx.Module):
    units = nodes.Param(int, default=16, min=1, description="Output features")
    use_bias = nodes.Param(bool, default=True, description="Add a learnable bias")

    def build(self, input_shape):
        features = input_shape[-1]
        # Glorot uniform, as the built-in Dense starts.
        limit = math.sqrt(6.0 / (features + self.units))
        self.add_parameter("weight", (cyx.Tensor.random([features, self.units]) * 2.0 - 1.0) * limit)
        if self.use_bias:
            self.add_parameter("bias", cyx.Tensor.zeros([self.units]))

    def output_shape(self, input_shape):
        return list(input_shape[:-1]) + [self.units]

    def forward(self, x):
        self.in_shape = x.shape()
        self.x2 = rows(x)
        y = matmul(self.x2, self.parameter("weight"))
        if self.use_bias:
            y = y + self.parameter("bias").broadcast_to(y.shape())
        return y.reshape(list(self.in_shape[:-1]) + [self.units])

    def backward(self, grad_output):
        g = rows(grad_output)
        self.set_gradient("weight", matmul(self.x2.transpose(), g))
        if self.use_bias:
            self.set_gradient("bias", g.sum(dim=0))
        return matmul(g, self.parameter("weight").transpose()).reshape(self.in_shape)


@nodes.node(id=ID, name="LayerNorm", version="1.0.0", category=CATEGORY, role="layer",
            display_name="Layer Norm (Python)",
            description="Normalise the last dimension to mean 0 and variance 1, then scale and shift")
class LayerNorm(cyx.Module):
    eps = nodes.Param(float, default=1e-5, min=1e-12, description="Added to the variance")

    def build(self, input_shape):
        features = input_shape[-1]
        self.add_parameter("gamma", cyx.Tensor.ones([features]))
        self.add_parameter("beta", cyx.Tensor.zeros([features]))

    def forward(self, x):
        self.in_shape = x.shape()
        x2 = rows(x)
        mean = x2.mean(dim=1, keepdim=True).broadcast_to(x2.shape())
        centred = x2 - mean
        variance = (centred * centred).mean(dim=1, keepdim=True)
        self.inv_std = (1.0 / (variance + self.eps).sqrt()).broadcast_to(x2.shape())
        self.x_hat = centred * self.inv_std
        gamma = self.parameter("gamma").broadcast_to(x2.shape())
        beta = self.parameter("beta").broadcast_to(x2.shape())
        return (self.x_hat * gamma + beta).reshape(self.in_shape)

    def backward(self, grad_output):
        g = rows(grad_output)
        features = g.shape()[1]
        self.set_gradient("gamma", (g * self.x_hat).sum(dim=0))
        self.set_gradient("beta", g.sum(dim=0))
        d_hat = g * self.parameter("gamma").broadcast_to(g.shape())
        sum_d = d_hat.sum(dim=1, keepdim=True).broadcast_to(g.shape())
        sum_dx = (d_hat * self.x_hat).sum(dim=1, keepdim=True).broadcast_to(g.shape())
        dx = (d_hat * float(features) - sum_d - self.x_hat * sum_dx) * self.inv_std / float(features)
        return dx.reshape(self.in_shape)


# ------------------------------------------------------------ activations
# Tier 1: f(x) and its derivative df(x); the node is generated from them.

@nodes.activation(id=ID, name="ReLU", version="1.0.0", category=CATEGORY,
                  display_name="ReLU (Python)", description="max(0, x)")
class ReLU:
    def f(self, x):
        return x * relu_mask(x)

    def df(self, x):
        return relu_mask(x)


@nodes.activation(id=ID, name="Sigmoid", version="1.0.0", category=CATEGORY,
                  display_name="Sigmoid (Python)", description="1 / (1 + e^-x)")
class Sigmoid:
    def f(self, x):
        return sigmoid(x)

    def df(self, x):
        s = sigmoid(x)
        return s * (1.0 - s)


@nodes.activation(id=ID, name="Tanh", version="1.0.0", category=CATEGORY,
                  display_name="Tanh (Python)", description="(e^x - e^-x) / (e^x + e^-x)")
class Tanh:
    def f(self, x):
        return tanh(x)

    def df(self, x):
        t = tanh(x)
        return 1.0 - t * t


@nodes.activation(id=ID, name="Softplus", version="1.0.0", category=CATEGORY,
                  display_name="Softplus (Python)", description="log(1 + e^x), computed without overflow")
class Softplus:
    def f(self, x):
        # max(x, 0) + log(1 + e^-|x|)
        return (x + x.abs()) * 0.5 + ((-x.abs()).exp() + 1.0).log()

    def df(self, x):
        return sigmoid(x)


@nodes.activation(id=ID, name="GELU", version="1.0.0", category=CATEGORY,
                  display_name="GELU (Python)",
                  description="0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), the built-in GELU's form")
class GELU:
    def f(self, x):
        inner = (x + x * x * x * 0.044715) * GELU_C
        return x * 0.5 * (tanh(inner) + 1.0)

    def df(self, x):
        inner = (x + x * x * x * 0.044715) * GELU_C
        t = tanh(inner)
        d_inner = (x * x * (3.0 * 0.044715) + 1.0) * GELU_C
        return (t + 1.0) * 0.5 + x * 0.5 * (1.0 - t * t) * d_inner


@nodes.activation(id=ID, name="SiLU", version="1.0.0", category=CATEGORY,
                  display_name="SiLU / Swish (Python)", description="x * sigmoid(x)")
class SiLU:
    def f(self, x):
        return x * sigmoid(x)

    def df(self, x):
        s = sigmoid(x)
        return s + x * s * (1.0 - s)


# Tier 2 activations: they have a setting.

@nodes.node(id=ID, name="LeakyReLU", version="1.0.0", category=CATEGORY, role="activation",
            display_name="Leaky ReLU (Python)", description="x if x > 0, else slope * x")
class LeakyReLU(cyx.Module):
    negative_slope = nodes.Param(float, default=0.01, min=0.0, description="Slope for x < 0")

    def forward(self, x):
        self.mask = relu_mask(x)
        return x * self.mask + x * (1.0 - self.mask) * self.negative_slope

    def backward(self, grad_output):
        return grad_output * (self.mask + (1.0 - self.mask) * self.negative_slope)


@nodes.node(id=ID, name="ELU", version="1.0.0", category=CATEGORY, role="activation",
            display_name="ELU (Python)", description="x if x > 0, else alpha (e^x - 1)")
class ELU(cyx.Module):
    alpha = nodes.Param(float, default=1.0, min=0.0, description="Scale of the negative part")

    def forward(self, x):
        self.mask = relu_mask(x)
        self.e = ((x - x.abs()) * 0.5).exp()  # e^min(x, 0): no overflow for large x
        return x * self.mask + (self.e - 1.0) * (1.0 - self.mask) * self.alpha

    def backward(self, grad_output):
        return grad_output * (self.mask + self.e * (1.0 - self.mask) * self.alpha)


# ----------------------------------------------------------------- losses
# A loss takes the model's predictions and the Data Input's targets and gives
# one value; backward gives dLoss/dpredictions with the mean scaling, as the
# built-in losses do. target says what the labels are: "categorical" (one-hot
# rows as wide as the model's output) or "continuous" (values).

@nodes.loss(id=ID, name="CrossEntropy", version="1.0.0", target="categorical", category=CATEGORY,
            display_name="Cross Entropy (Python)",
            description="Softmax over the last layer's outputs, then the mean of -log p(class)")
class CrossEntropy(cyx.Loss):
    def forward(self, predictions, targets):
        log_p = softmax(predictions).clip(1e-12, 1.0).log()
        return (targets * log_p).sum() * (-1.0 / row_count(predictions))

    def backward(self, predictions, targets):
        return (softmax(predictions) - targets) * (1.0 / row_count(predictions))


@nodes.loss(id=ID, name="MSE", version="1.0.0", target="continuous", category=CATEGORY,
            display_name="MSE (Python)", description="mean((prediction - target)^2)")
class MSE(cyx.Loss):
    def forward(self, predictions, targets):
        d = predictions - targets
        return (d * d).mean()

    def backward(self, predictions, targets):
        return (predictions - targets) * (2.0 / predictions.num_elements())


@nodes.loss(id=ID, name="L1", version="1.0.0", target="continuous", category=CATEGORY,
            display_name="L1 (Python)", description="mean(|prediction - target|)")
class L1(cyx.Loss):
    def forward(self, predictions, targets):
        return (predictions - targets).abs().mean()

    def backward(self, predictions, targets):
        return (predictions - targets).sign() * (1.0 / predictions.num_elements())


@nodes.loss(id=ID, name="Huber", version="1.0.0", target="continuous", category=CATEGORY,
            display_name="Huber (Python)",
            description="Squared error below delta, linear above: less pull from outliers than MSE")
class Huber(cyx.Loss):
    delta = nodes.Param(float, default=1.0, min=1e-6, description="Where the loss turns from squared to linear")

    def forward(self, predictions, targets):
        d = predictions - targets
        a = d.abs()
        small = cyx.Tensor.ones(a.shape()) * (a < self.delta)
        squared = d * d * 0.5
        linear = (a - 0.5 * self.delta) * self.delta
        return (squared * small + linear * (1.0 - small)).mean()

    def backward(self, predictions, targets):
        d = predictions - targets
        return d.clip(-self.delta, self.delta) * (1.0 / predictions.num_elements())


ALL = [Dense, LayerNorm, ReLU, LeakyReLU, ELU, Sigmoid, Tanh, Softplus, GELU, SiLU,
       CrossEntropy, MSE, L1, Huber]
