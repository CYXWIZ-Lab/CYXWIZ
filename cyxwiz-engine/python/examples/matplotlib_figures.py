"""Figures in the Engine with matplotlib.

Run in the Script Editor (F5): each plt.show() puts the open figures in the
Plot Output window. In a notebook cell they show under the cell.
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 200)

# 1. Line plot with two series
plt.figure(figsize=(8, 4))
plt.plot(x, np.sin(x), label="sin")
plt.plot(x, np.cos(x), label="cos")
plt.title("Line plot")
plt.xlabel("x")
plt.legend()
plt.show()

# 2. Scatter plot
rng = np.random.default_rng(0)
a = rng.normal(size=300)
b = 0.6 * a + rng.normal(scale=0.5, size=300)
plt.figure(figsize=(6, 5))
plt.scatter(a, b, alpha=0.6)
plt.title("Scatter plot")
plt.show()

# 3. Histogram
plt.figure(figsize=(6, 4))
plt.hist(rng.normal(size=5000), bins=40)
plt.title("Histogram")
plt.show()

print("3 figures sent to Plot Output")
