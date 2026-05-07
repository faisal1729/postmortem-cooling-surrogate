
"""
Chapter 2: Behaviour of the Marshall–Hoare Model

This script demonstrates:
- Newton cooling model
- Marshall–Hoare model and derivative
- Asymptotic behaviour
- Parameter sensitivity (A, B, Ta)

No external data is required.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------
# Setup output directory
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "mh_plots"
OUT_DIR.mkdir(exist_ok=True)


# -------------------------------
# Models
# -------------------------------

def T_MH(t, A, B, T0, Ta):
    Delta = T0 - Ta
    comp1 = A * np.exp(B * t)
    comp2 = (1.0 - A) * np.exp((A * B / (A - 1.0)) * t)
    return Ta + Delta * (comp1 + comp2)


def dTdt_MH(t, A, B, T0, Ta):
    Delta = T0 - Ta
    pref = Delta * A * B
    return pref * (np.exp(B * t) - np.exp((A * B / (A - 1.0)) * t))


def marshall_hoare(t, A, B, T0, Ta):
    r1 = np.clip(B * t, -709, 709)
    delta = A - 1.0
    delta_safe = np.where(np.abs(delta) < 1e-6, 1.0, delta)

    x = np.clip((B * t) / delta_safe, -700, 709)
    second = np.where(np.abs(delta) < 1e-6, 0.0, (1.0 - A) * np.exp(x))

    theta = np.exp(r1) * (A + second)
    return Ta + (T0 - Ta) * theta


# -------------------------------
# Utility
# -------------------------------

def save_plot(fig, name):
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


# -------------------------------
# Plots
# -------------------------------

def plot_newton():
    T0, Ta, B = 37, 22, -0.106
    t = np.linspace(0, 60, 500)

    T = Ta + (T0 - Ta) * np.exp(B * t)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t, T, label="Newton cooling")
    ax.axhline(Ta, linestyle="--", label="Ambient temperature")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Temperature (°C)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    save_plot(fig, "newton_cooling")


def plot_mh_and_derivative():
    A, B, T0, Ta = 2.75, -0.108, 37.0, 20.0
    t = np.linspace(0, 48, 1000)

    T = T_MH(t, A, B, T0, Ta)
    dT = dTdt_MH(t, A, B, T0, Ta)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(t, T, label="T(t)")
    ax1.set_xlabel("Time (hours)")
    ax1.set_ylabel("Temperature")

    ax2 = ax1.twinx()
    ax2.plot(t, dT, "--", label="dT/dt")

    ax1.grid(True)
    fig.tight_layout()

    save_plot(fig, "mh_with_derivative")


def plot_asymptotic():
    A, B, T0, Ta = 2.75, -0.108, 37.0, 20.0
    Delta = T0 - Ta
    lambda2 = A * B / (A - 1.0)

    t = np.linspace(0, 80, 1000)

    T = Ta + Delta * (A*np.exp(B*t) + (1-A)*np.exp(lambda2*t))
    asymp = np.abs(Delta * A) * np.exp(B*t)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.semilogy(t, np.abs(T - Ta), label="|T - Ta|")
    ax.semilogy(t, asymp, "--", label="Asymptotic")
    ax.set_xlabel("Time")
    ax.set_ylabel("Log scale")
    ax.legend()
    ax.grid(True)

    save_plot(fig, "mh_asymptotic")


def plot_vary_A():
    t = np.linspace(0, 60, 400)
    fig, ax = plt.subplots()

    for A in [1, 2, 5, 15]:
        T = marshall_hoare(t, A, -0.106, 37, 22)
        ax.plot(t, T, label=f"A={A}")

    ax.legend()
    ax.grid(True)
    save_plot(fig, "mh_varyA")


def plot_vary_B():
    t = np.linspace(0, 60, 400)
    fig, ax = plt.subplots()

    for B in [-0.05, -0.106, -0.2]:
        T = marshall_hoare(t, 2.0, B, 37, 22)
        ax.plot(t, T, label=f"B={B}")

    ax.legend()
    ax.grid(True)
    save_plot(fig, "mh_varyB")


# -------------------------------
# Main
# -------------------------------

def main():
    plot_newton()
    plot_mh_and_derivative()
    plot_asymptotic()
    plot_vary_A()
    plot_vary_B()

    print(f"All Chapter 2 plots saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()
