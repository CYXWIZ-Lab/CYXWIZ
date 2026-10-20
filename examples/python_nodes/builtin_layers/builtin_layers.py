# SPDX-License-Identifier: Apache-2.0
# Part of the CyxWiz plugin SDK examples (see cmake/sdk/NOTICE).
"""Ten CyxWiz layers written again as Python nodes.

Each node does what the built-in node of the same name does, written only
with pycyxwiz tensor operations, so you can read how a layer is made and copy
it. Put them between a Data Input and a loss like the built-in ones; the
example graph blobs_python_layers.cyxgraph does.

    Dense      y = x W + b, learnable W [in, units] and b [units]
    LayerNorm  normalise the last dimension, learnable gain and shift
    ReLU, LeakyReLU, ELU, Sigmoid, Tanh, Softplus, GELU (tanh form), SiLU

Losses are not here: a Python node is one tensor in and one tensor out, so
use the built-in loss nodes.

Bundle them (writes nodes/builtin-layers.cyxnode/):

    from pycyxwiz import nodes
    import builtin_layers as b
    nodes.bundle(b.Dense, b.LayerNorm, b.ReLU, b.LeakyReLU, b.ELU, b.Sigmoid,
                 b.Tanh, b.Softplus, b.GELU, b.SiLU, folder="nodes",
                 name="builtin-layers")
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
