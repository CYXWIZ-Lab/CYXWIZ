#!/usr/bin/env python3
# PyCyxWiz version of the Coursera Machine Learning ex1 exercise (linear regression).

import sys
import os
from pathlib import Path
from typing import List, Tuple

def configure_pycyxwiz_paths():
    repo_root = Path(__file__).resolve().parents[2]
    if sys.platform != "win32":
        return
    candidates = [
        repo_root / "build" / "windows-release" / "bin" / "Release",
        repo_root / "build" / "windows-release" / "lib" / "Release",
    ]
    for candidate in candidates:
        if candidate.exists():
            os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")
    lib_path = repo_root / "build" / "windows-release" / "lib" / "Release"
    if lib_path.exists() and str(lib_path) not in sys.path:
        sys.path.insert(0, str(lib_path))

configure_pycyxwiz_paths()

import numpy as np
import matplotlib.pyplot as plt
import pycyxwiz as cx
import pycyxwiz.linalg as la

DATAFILE = Path(__file__).resolve().parent / "ex1data1.txt"

def warmup_identity() -> np.ndarray:
    # Return a 5x5 identity matrix using PyCyxWiz math.
    return la.eye(5)

def load_data(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", dtype=np.float64)

def plot_data(X: np.ndarray, y: np.ndarray) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 1], y[:, 0], color="red", marker="x", s=60, label="Training data")
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.title("Training data for linear regression")
    plt.grid(True)
    plt.legend()

def compute_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    m = X.shape[0]
    predictions = la.matmul(X, theta)
    errors = predictions - y
    cost_matrix = la.matmul(la.transpose(errors), errors)
    return float(cost_matrix.item()) / (2 * m)

def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    alpha: float,
    iterations: int,
) -> Tuple[np.ndarray, List[float]]:
    m = X.shape[0]
    theta_history = theta.copy()
    cost_history: List[float] = []
    for _ in range(iterations):
        predictions = la.matmul(X, theta_history)
        errors = predictions - y
        gradient = la.matmul(la.transpose(X), errors) / m
        theta_history = theta_history - alpha * gradient
        cost_history.append(compute_cost(X, y, theta_history))
    return theta_history, cost_history

def plot_linear_fit(X: np.ndarray, theta: np.ndarray) -> None:
    x_vals = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
    X_fit = np.column_stack((np.ones_like(x_vals), x_vals))
    y_fit = la.matmul(X_fit, theta)
    plt.plot(x_vals, y_fit.flatten(), label="Linear regression", color="blue", linewidth=2)

def plot_cost_surface(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> None:
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)
    J_vals = np.zeros((theta0_vals.size, theta1_vals.size))
    for i, t0 in enumerate(theta0_vals):
        for j, t1 in enumerate(theta1_vals):
            t = np.array([[t0], [t1]], dtype=np.float64)
            J_vals[i, j] = compute_cost(X, y, t)
    theta0_grid, theta1_grid = np.meshgrid(theta0_vals, theta1_vals)

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(theta0_grid, theta1_grid, J_vals.T, cmap="viridis", edgecolor="none", alpha=0.9)
    ax1.set_title("Surface of Cost Function")
    ax1.set_xlabel(r"$\theta_0$")
    ax1.set_ylabel(r"$\theta_1$")
    ax1.set_zlabel("Cost J(θ)")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.contour(theta0_grid, theta1_grid, J_vals.T, levels=np.logspace(-2, 3, 20), cmap="plasma")
    ax2.scatter(theta[0, 0], theta[1, 0], color="red", marker="x", s=80, label="Gradient descent minimum")
    ax2.set_title("Contour of Cost Function")
    ax2.set_xlabel(r"$\theta_0$")
    ax2.set_ylabel(r"$\theta_1$")
    ax2.legend()
    plt.tight_layout()

def predict(theta: np.ndarray, population: float) -> float:
    features = np.array([1.0, population], dtype=np.float64)
    return float(np.dot(features, theta).item())

def main() -> None:
    os.makedirs(DATAFILE.parent, exist_ok=True)
    print("Warm-up identity matrix:")
    print(warmup_identity())

    print("\nLoading data...")
    data = load_data(DATAFILE)
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    m = len(y)
    X_b = np.hstack((np.ones((m, 1), dtype=np.float64), X))

    plot_data(X_b, y)

    theta = np.zeros((2, 1), dtype=np.float64)
    print("Initial cost (theta zeros):", compute_cost(X_b, y, theta))

    theta_final, cost_history = gradient_descent(X_b, y, theta, alpha=0.01, iterations=1500)
    print("Theta found by gradient descent:", theta_final.flatten())

    plot_linear_fit(X_b, theta_final)
    plt.legend()

    for pop in [3.5, 7.0]:
        profit_prediction = predict(theta_final, pop)
        print(f"Predicted profit for population {int(pop * 10000):,d}: ${profit_prediction * 10000:,.2f}")

    plot_cost_surface(X_b, y, theta_final)
    plt.show()

if __name__ == "__main__":
    cx.initialize()
    try:
        main()
    finally:
        cx.shutdown()
