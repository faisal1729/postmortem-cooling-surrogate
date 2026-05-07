"""
Chapter 3: Kernel Analysis for Gaussian Process Surrogate Modelling

This script:
- compares Squared Exponential and Matern kernels
- visualizes prior samples
- performs GP regression on a single cooling curve
- evaluates uncertainty behaviour
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import gpflow

from pathlib import Path
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent.parent

curve_path = BASE_DIR / "data" / "sample_curve.gnu"

RESULTS_DIR = BASE_DIR / "results" / "chapter3"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)
tf.random.set_seed(42)

if not curve_path.exists():
    raise FileNotFoundError(
        f"Cooling curve not found at:\n{curve_path}"
    )

curve = pd.read_csv(
    curve_path,
    sep=r"\s+",
    comment="#",
    header=None,
    names=["time", "temperature"]
)

curve["time"] = curve["time"] / 60.0

X = curve["time"].values.reshape(-1, 1)
Y = curve["temperature"].values.reshape(-1, 1)

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(X)
Y_scaled = y_scaler.fit_transform(Y)

def save_plot(fig, name):
    png_path = RESULTS_DIR / f"{name}.png"
    pdf_path = RESULTS_DIR / f"{name}.pdf"

    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    fig.savefig(pdf_path, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved: {png_path.name}")

def plot_cooling_curve():
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(X, Y, linewidth=2)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Cooling Curve Example")
    ax.grid(True)

    fig.tight_layout()

    save_plot(fig, "cooling_curve")

def plot_prior_samples():

    kernels = {
        "Squared Exponential":
            gpflow.kernels.SquaredExponential(lengthscales=1.0),

        "Matern 3/2":
            gpflow.kernels.Matern32(lengthscales=1.0),

        "Matern 5/2":
            gpflow.kernels.Matern52(lengthscales=1.0),
    }

    Xtest = np.linspace(-3, 3, 200).reshape(-1, 1)

    for kernel_name, kernel in kernels.items():

        cov = kernel(Xtest).numpy()

        samples = np.random.multivariate_normal(
            mean=np.zeros(len(Xtest)),
            cov=cov,
            size=5
        )

        fig, ax = plt.subplots(figsize=(9, 5))

        for sample in samples:
            ax.plot(Xtest, sample)

        ax.set_title(f"{kernel_name} Prior Samples")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True)

        fig.tight_layout()

        safe_name = kernel_name.lower().replace(" ", "_").replace("/", "_")
        save_plot(fig, f"prior_{safe_name}")

def fit_gp(kernel, kernel_name):

    model = gpflow.models.GPR(
        data=(X_scaled, Y_scaled),
        kernel=kernel,
        mean_function=None
    )

    optimizer = gpflow.optimizers.Scipy()

    optimizer.minimize(
        model.training_loss,
        model.trainable_variables
    )

    Xtest = np.linspace(
        X_scaled.min() - 1,
        X_scaled.max() + 1,
        500
    ).reshape(-1, 1)

    mean, var = model.predict_f(Xtest)

    mean = mean.numpy()
    std = np.sqrt(var.numpy())

    Xtest_original = x_scaler.inverse_transform(Xtest)
    mean_original = y_scaler.inverse_transform(mean)

    upper = y_scaler.inverse_transform(mean + 2 * std)
    lower = y_scaler.inverse_transform(mean - 2 * std)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        X,
        Y,
        "k.",
        markersize=5,
        label="Observed data"
    )

    ax.plot(
        Xtest_original,
        mean_original,
        linewidth=2,
        label="GP mean"
    )

    ax.fill_between(
        Xtest_original.flatten(),
        lower.flatten(),
        upper.flatten(),
        alpha=0.25,
        label="95% confidence interval"
    )

    ax.set_title(f"Gaussian Process Regression ({kernel_name})")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()

    safe_name = kernel_name.lower().replace(" ", "_").replace("/", "_")
    save_plot(fig, f"gp_fit_{safe_name}")

    print("\n------------------------------------")
    print(kernel_name)
    print("------------------------------------")
    print(gpflow.utilities.print_summary(model))

def main():

    print("Running Chapter 3 kernel analysis...")

    plot_cooling_curve()

    plot_prior_samples()

    fit_gp(
        gpflow.kernels.SquaredExponential(),
        "Squared Exponential"
    )

    fit_gp(
        gpflow.kernels.Matern32(),
        "Matern 3/2"
    )

    fit_gp(
        gpflow.kernels.Matern52(),
        "Matern 5/2"
    )

    print("\nAll Chapter 3 results saved to:")
    print(RESULTS_DIR.resolve())

if __name__ == "__main__":
    main()
