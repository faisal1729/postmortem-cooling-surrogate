
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

# Output directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

OUT_DIR = BASE_DIR / "results" / "chapter2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


"""We first plot Newton's Cooling model"""

# Parameters
T0 = 37          # Initial body temperature
Ta = 22          # Ambient temperature
B  = -0.106      # Cooling rate
A  = 2.75


t = np.linspace(0, 60, 500)

# Newton cooling model
T_newton = Ta + (T0 - Ta) * np.exp(B * t)

plt.figure(figsize=(8,4.5))

plt.plot(t, T_newton,
         linewidth=2,
         label=r"Newton cooling ($B=-0.106$)")

# Ambient temperature line
plt.axhline(Ta,
            linestyle="--",
            linewidth=1.5,
            label=r"Ambient temperature $T_a=22^\circ$C")

plt.xlabel("Time")
plt.ylabel(r"Temperature ($^\circ$C)")
plt.grid(True)

plt.legend(frameon=True)
plt.tight_layout()

plt.savefig(OUT_DIR / "newton_cooling.png", dpi=300)
plt.close()

def marshall_hoare(t, A, B, T0, Ta):
    r1 = np.clip(B * t, -709, 709)
    delta = A - 1.0

    # Regularization near A = 1
    near_one = np.abs(delta) < 1e-6
    delta_safe = np.where(near_one, 1.0, delta)

    x = np.clip((B * t) / delta_safe, -700, 709)
    second = np.where(near_one, 0.0, (1.0 - A) * np.exp(x))

    theta = np.exp(r1) * (A + second)
    return Ta + (T0 - Ta) * theta

def dTdt_MH(t, A, B, T0, Ta):
    """
    Simplified derivative:
    dT/dt = (T0 - Ta) * A * B * ( exp(B t) - exp( (A B/(A-1)) t ) )
    """
    Delta = T0 - Ta
    pref = Delta * A * B
    e1 = np.exp(B * t)
    e2 = np.exp((A * B / (A - 1.0)) * t)
    return pref * (e1 - e2)

# --- Time grid ---
t_max = 48.0      # hours
t = np.linspace(0.0, t_max, 1000)

# --- Evaluate
T = marshall_hoare(t, A, B, T0, Ta)
dT = dTdt_MH(t, A, B, T0, Ta)

# --- Plotting
fig, ax1 = plt.subplots(figsize=(10,6))

ax1.plot(t, T, label=r'$T_{\mathrm{MH}}(t)$', linewidth=2)
ax1.set_xlabel('time (hours)')
ax1.set_ylabel('Temperature', fontsize=10)
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(t, dT, linestyle='--', label=r'$dT/dt$', linewidth=2)
ax2.set_ylabel('dT/dt (rate)', fontsize=10)

# Legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left')

#plt.title('Marshall–Hoare model and its time derivative')
plt.tight_layout()
fig.savefig(OUT_DIR / "mh_derivative.png",
            dpi=300,
            bbox_inches="tight")
plt.close()

"""## Asymptotic behaviour:"""

Delta = T0 - Ta
lambda2 = A * B / (A - 1.0)

# Time grid
t = np.linspace(0, 80, 1000)

# MH model
T = marshall_hoare(t, A, B, T0, Ta)
# Plot
plt.figure(figsize=(10,6))
plt.plot(t, T, label=r"$T_{\mathrm{MH}}(t)$", linewidth=2)
plt.axhline(Ta, linestyle="--", linewidth=1.5, label=r"$T_a$")

plt.xlabel("time")
plt.ylabel("Temperature")
plt.title("Convergence of $T_{MH}(t)$ to $T_a$")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# MH model
T = marshall_hoare(t, A, B, T0, Ta)
abs_T_minus_Ta = np.abs(T - Ta)

# Asymptotic approximation
asymp = np.abs(Delta * A) * np.exp(B*t)

# Plot
plt.figure(figsize=(10,6))
plt.semilogy(t, abs_T_minus_Ta, label=r"$|T(t)-T_a|$", linewidth=2)
plt.semilogy(t, asymp, "--", label=r"Asymptotic: $|\Delta A| e^{Bt}$", linewidth=2)

plt.xlabel("time")
plt.ylabel(r"$|T(t)-T_a|$ (log scale)")
plt.title(r"Semilog plot showing $T(t)-T_a \sim \Delta A e^{Bt}$")
plt.grid(alpha=0.3, which='both')
plt.legend()
plt.tight_layout()
plt.close()

# Helper to save figures
def save_plot(fig, name):
    pdf_path = OUT_DIR / f"{name}.pdf"
    png_path = OUT_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    print("Saved:", pdf_path, "and", png_path)

# Plot 1: Vary A
def plot_vary_A():
    T0 = 37.0
    Ta = 22.0
    B = -0.106
    A_values = [1, 2, 5, 15, 250]
    t = np.linspace(0, 60, 400)

    fig, ax = plt.subplots(figsize=(11, 6))
    for A in A_values:
        T = marshall_hoare(t, A, B, T0, Ta)
        ax.plot(t, T, label=f"A = {A}")

    ax.set_title("Marshall–Hoare Model — varying A")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True)
    save_plot(fig, "mh_varyA")
    plt.close(fig)

# Plot 2: Vary B
def plot_vary_B():
    T0 = 37.0
    Ta = 22.0
    A = 2.0
    B_values = [-0.05, -0.106, -0.2, -0.5]
    t = np.linspace(0, 60, 400)

    fig, ax = plt.subplots(figsize=(11, 6))
    for B in B_values:
        T = marshall_hoare(t, A, B, T0, Ta)
        ax.plot(t, T, label=f"B = {B}")

    ax.set_title("Marshall–Hoare Model — varying B")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True)
    save_plot(fig, "mh_varyB")
    plt.close(fig)

# Plot 3: Vary Ta
def plot_vary_Ta():
    T0 = 37.0
    A = 2.0
    B = -0.106
    Ta_values = [10, 18, 22, 28]
    t = np.linspace(0, 60, 400)

    fig, ax = plt.subplots(figsize=(11, 6))
    for Ta in Ta_values:
        T = marshall_hoare(t, A, B, T0, Ta)
        ax.plot(t, T, label=f"$T_a = {Ta}$°C")

    ax.set_title("Marshall–Hoare Model — varying ambient temperature")
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Temperature (°C)")
    ax.legend()
    ax.grid(True)
    save_plot(fig, "mh_varyTa")
    plt.close(fig)

# Main
if __name__ == "__main__":
    plot_vary_A()
    plot_vary_B()
    plot_vary_Ta()
    print("All plots saved to:", OUT_DIR.resolve())
