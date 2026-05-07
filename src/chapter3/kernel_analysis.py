

import gpflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Repository root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data file
file_path = BASE_DIR / "data" / "sample_curve.gnu"

# Output directory
RESULTS_DIR = BASE_DIR / "results" / "chapter3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

"""
Chapter 3 — Kernel Analysis: SE & Matérn Family
================================================
GPflow implementation for all visualisation figures in Chapter 3.
Each cell is labeled with the thesis figure it produces.

Requirements: gpflow, tensorflow, numpy, matplotlib, pandas
Data: one simulated FE cooling curve (coolingCurve1.gnu)
"""

# ============================================================
# CELL 0: Imports and data loading
# ============================================================

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load simulated cooling curve
df = pd.read_csv(file_path, sep=r"\s+", comment='#', header=None)
df.columns = ['time', 'temperature']

print(f"Loaded {len(df)} data points.")
print(f"Time range: [{df['time'].min():.1f}, {df['time'].max():.1f}] minutes")
print(f"Temp range: [{df['temperature'].min():.2f}, {df['temperature'].max():.2f}] °C")

# ============================================================
# CELL 1: Select 5 representative training points via KMeans
# ============================================================

n_train = 5

X_k = df[["time", "temperature"]].to_numpy()
X_k_norm = (X_k - X_k.mean(axis=0)) / X_k.std(axis=0)
kmeans = KMeans(n_clusters=n_train, random_state=42, n_init=10).fit(X_k_norm)

selected_indices = []
for c_norm in kmeans.cluster_centers_:
    dists = np.linalg.norm(X_k_norm - c_norm, axis=1)
    selected_indices.append(int(np.argmin(dists)))
selected_indices = np.sort(np.unique(selected_indices))

if len(selected_indices) < n_train:
    selected_indices = np.linspace(0, len(df) - 1, n_train, dtype=int)

X_train = df["time"].iloc[selected_indices].to_numpy().reshape(-1, 1)
y_train = df["temperature"].iloc[selected_indices].to_numpy().reshape(-1, 1)

X_all = df["time"].to_numpy().reshape(-1, 1)
y_all = df["temperature"].to_numpy().reshape(-1, 1)

print(f"Training points ({len(selected_indices)}):")
for i in range(len(X_train)):
    print(f"  t = {X_train[i,0]:.1f} min, T = {y_train[i,0]:.2f} °C")

# ============================================================
# CELL 2: Standardization (using full data range for stability)
# ============================================================
# Key fix: scale using full data range, not just the 3 training
# points. This avoids the poor moment estimates that caused the
# earlier 3-point failure.
X_mean, X_std = X_all.mean(), X_all.std()
y_mean, y_std = y_all.mean(), y_all.std()

def scale_X(X):
    return (X - X_mean) / X_std

def scale_y(y):
    return (y - y_mean) / y_std

def unscale_y(y_s):
    return y_s * y_std + y_mean

X_train_s = scale_X(X_train)
y_train_s = scale_y(y_train)
X_all_s = scale_X(X_all)

print(f"Scaling (from full data):")
print(f"  X_mean={X_mean:.2f}, X_std={X_std:.2f}")
print(f"  y_mean={y_mean:.2f}, y_std={y_std:.2f}")

# ============================================================
# CELL 3: Helper — build, optimize, and predict with a GPflow model
# ============================================================

def fit_gp(kernel, X_tr, y_tr, X_pred, n_restarts=10):
    """
    Fit a GPflow GPR model and return predictions in original scale.
    Uses a trainable constant mean function and near-zero likelihood
    variance (deterministic simulator).
    """
    best_lml = -np.inf
    best_model = None

    for i in range(n_restarts):
        k = gpflow.utilities.deepcopy(kernel)
        m = gpflow.models.GPR(
            data=(X_tr.astype(np.float64), y_tr.astype(np.float64)),
            kernel=k,
            mean_function=None
        )
        # Fix likelihood variance to near-zero (deterministic simulator)
        m.likelihood.variance = gpflow.Parameter(
            1e-6, transform=gpflow.utilities.positive(), trainable=False
        )

        # Randomize kernel hyperparameters for restart (except first)
        if i > 0:
            for p in m.trainable_parameters:
                unc = p.unconstrained_variable.numpy()
                unc += np.random.randn(*unc.shape) * 1.5
                p.unconstrained_variable.assign(unc)

        opt = gpflow.optimizers.Scipy()
        try:
            opt.minimize(m.training_loss, m.trainable_variables,
                         options=dict(maxiter=2000))
        except Exception:
            continue

        lml = m.log_marginal_likelihood().numpy()
        if lml > best_lml:
            best_lml = lml
            best_model = m

    if best_model is None:
        raise RuntimeError("All optimization restarts failed.")

    # Print learned hyperparameters
    print(f"  variance    = {best_model.kernel.variance.numpy():.4f}")
    print(f"  lengthscale = {best_model.kernel.lengthscales.numpy():.4f}")
    print(f"  LML = {best_lml:.4f}")

    # Predict
    f_mean, f_var = best_model.predict_f(X_pred.astype(np.float64))
    f_mean = f_mean.numpy().flatten()
    f_std = np.sqrt(np.maximum(f_var.numpy().flatten(), 0.0))

    mean_orig = unscale_y(f_mean)
    lower_orig = unscale_y(f_mean - 1.96 * f_std)
    upper_orig = unscale_y(f_mean + 1.96 * f_std)

    return mean_orig, lower_orig, upper_orig, best_model

# ============================================================
# CELL 4: Helper — generic plotting function
# ============================================================
def plot_gp_fit(time, temp, X_tr, y_tr, mean, lower, upper,
                title="", filename=None):
    """Plot GP posterior with data, training points, mean, and 95% band."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, temp, color='red', linewidth=1.2, label="Simulated data")
    ax.scatter(X_tr.flatten(), y_tr.flatten(), marker='o', s=60,
               color='black', zorder=5, label='Training points')
    ax.plot(time, mean, color='blue', linewidth=1.5, label="GP mean")
    ax.fill_between(time, lower, upper, alpha=0.25, color='blue',
                    label="95% credible band")
    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if filename:
        fig.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"Saved: {filename}")
    plt.show()

# ============================================================
# CELL 5: Figure 3.1 — GP prior samples with SE kernel
#          (fig:rbf-samples)
# ============================================================
# Draw prior samples on a fine grid in [0, 1]
n_grid = 200
X_grid = np.linspace(0, 1, n_grid).reshape(-1, 1)

k_se_prior = gpflow.kernels.SquaredExponential(
    variance=1.0, lengthscales=0.3
)
K = k_se_prior.K(X_grid).numpy()
K += 1e-8 * np.eye(n_grid)  # jitter for numerical Cholesky
L = np.linalg.cholesky(K)

fig, ax = plt.subplots(figsize=(8, 5))
for i in range(5):
    z = np.random.randn(n_grid)
    sample = L @ z
    ax.plot(X_grid.flatten(), sample, linewidth=1.2)
ax.set_xlabel("$x$", fontsize=12)
ax.set_ylabel("$f(x)$", fontsize=12)
ax.set_title("Prior samples: SE kernel ($\\sigma_f^2=1$, $\\ell=0.3$)",
             fontsize=13)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "rbf_samples.png",
            dpi=200, bbox_inches='tight')
print("Saved: rbf_samples.png")
plt.show()

# ============================================================
# CELL 6: Figure 3.2 — GP prior samples with Matérn kernels
#          (fig:matern-samples)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
nu_values = [0.5, 1.5, 2.5]
labels = [r"$\nu = 1/2$", r"$\nu = 3/2$", r"$\nu = 5/2$"]

for ax, nu, label in zip(axes, nu_values, labels):
    k_mat = gpflow.kernels.Matern12(variance=1.0, lengthscales=0.3) if nu == 0.5 \
       else gpflow.kernels.Matern32(variance=1.0, lengthscales=0.3) if nu == 1.5 \
       else gpflow.kernels.Matern52(variance=1.0, lengthscales=0.3)
    K = k_mat.K(X_grid).numpy() + 1e-8 * np.eye(n_grid)
    L = np.linalg.cholesky(K)
    for i in range(5):
        sample = L @ np.random.randn(n_grid)
        ax.plot(X_grid.flatten(), sample, linewidth=1.0)
    ax.set_title(label, fontsize=12)
    ax.set_xlabel("$x$", fontsize=11)
    ax.grid(True, alpha=0.3)
axes[0].set_ylabel("$f(x)$", fontsize=11)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "matern_samples.png",
            dpi=200, bbox_inches='tight')
print("Saved: matern_samples.png")
plt.show()

# ============================================================
# CELL 7: Figure 3.3 — Kernel shape comparison k(r)
#          (fig:kernel-shapes)
# ============================================================
r = np.linspace(0, 0.6, 500).reshape(-1, 1)
origin = np.zeros((1, 1))

kernels_plot = {
    "SE":              gpflow.kernels.SquaredExponential(variance=1.0, lengthscales=0.1),
    r"Matérn $\nu=1/2$": gpflow.kernels.Matern12(variance=1.0, lengthscales=0.1),
    r"Matérn $\nu=3/2$": gpflow.kernels.Matern32(variance=1.0, lengthscales=0.1),
    r"Matérn $\nu=5/2$": gpflow.kernels.Matern52(variance=1.0, lengthscales=0.1),
}

fig, ax = plt.subplots(figsize=(8, 5))
for name, k in kernels_plot.items():
    # k(r) = k(0, r) since stationary
    kr = k.K(origin, r).numpy().flatten()
    ax.plot(r.flatten(), kr, linewidth=1.5, label=name)
ax.set_xlabel("$r = \\|\\mathbf{x} - \\mathbf{x}'\\|$", fontsize=12)
ax.set_ylabel("$k(r)$", fontsize=12)
ax.set_title("Covariance functions ($\\ell = 0.1$, $\\sigma_f^2 = 1$)",
             fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "kernel_shapes.png",
            dpi=200, bbox_inches='tight')
print("Saved: kernel_shapes.png")
plt.show()

# ============================================================
# GP posterior with SE kernel — 3 training points
# Produces: fig:gp-posterior-rbf (gp_posterior_rbf.png)
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# --- Select 3 training points manually ---
target_times = [200, 700, 1200]
selected_indices = [np.argmin(np.abs(df["time"].values - t)) for t in target_times]

X_train = df["time"].iloc[selected_indices].to_numpy().reshape(-1, 1)
y_train = df["temperature"].iloc[selected_indices].to_numpy().reshape(-1, 1)
X_all = df["time"].to_numpy().reshape(-1, 1)

print(f"Training points ({len(selected_indices)}):")
for i in range(len(X_train)):
    print(f"  t = {X_train[i,0]:.1f} min, T = {y_train[i,0]:.2f} °C")

# --- Scale using full data range ---
X_mean, X_std = X_all.mean(), X_all.std()
y_all_np = df["temperature"].to_numpy().reshape(-1, 1)
y_mean, y_std = y_all_np.mean(), y_all_np.std()

X_train_s = (X_train - X_mean) / X_std
y_train_s = (y_train - y_mean) / y_std
X_all_s = (X_all - X_mean) / X_std

# --- Fit SE kernel with multiple restarts ---
best_lml = -np.inf
best_model = None

for i in range(10):
    k = gpflow.kernels.SquaredExponential()
    m = gpflow.models.GPR(
        data=(X_train_s.astype(np.float64), y_train_s.astype(np.float64)),
        kernel=k,
        mean_function=None
    )
    m.likelihood.variance = gpflow.Parameter(
        1e-6, transform=gpflow.utilities.positive(), trainable=False
    )
    if i > 0:
        for p in m.trainable_parameters:
            unc = p.unconstrained_variable.numpy()
            unc += np.random.randn(*unc.shape) * 1.5
            p.unconstrained_variable.assign(unc)

    opt = gpflow.optimizers.Scipy()
    try:
        opt.minimize(m.training_loss, m.trainable_variables,
                     options=dict(maxiter=2000))
    except Exception:
        continue

    lml = m.log_marginal_likelihood().numpy()
    if lml > best_lml:
        best_lml = lml
        best_model = m

# --- Predict and unscale ---
f_mean, f_var = best_model.predict_f(X_all_s.astype(np.float64))
f_mean = f_mean.numpy().flatten()
f_std = np.sqrt(np.maximum(f_var.numpy().flatten(), 0.0))

mean_orig = f_mean * y_std + y_mean
lower_orig = (f_mean - 1.96 * f_std) * y_std + y_mean
upper_orig = (f_mean + 1.96 * f_std) * y_std + y_mean

print(f"SE kernel optimized:")
print(f"  variance    = {best_model.kernel.variance.numpy():.4f}")
print(f"  lengthscale = {best_model.kernel.lengthscales.numpy():.4f}")
print(f"  LML = {best_lml:.4f}")

# --- Enhanced plot for thesis ---
fig, ax = plt.subplots(figsize=(10, 5.5))

# Simulated data as thin grey line
ax.plot(df["time"], df["temperature"], color='grey', linewidth=1.0,
        alpha=0.7, linestyle='-', label="Simulated cooling curve")

# 95% credible band
ax.fill_between(df["time"].values, lower_orig, upper_orig,
                alpha=0.20, color='#4A90D9', label=r"95% credible band")

# GP mean
ax.plot(df["time"], mean_orig, color='#2C5F8A', linewidth=2.0,
        label="GP posterior mean")

# Training points
ax.scatter(X_train.flatten(), y_train.flatten(), marker='o', s=80,
           facecolors='black', edgecolors='white', linewidths=1.2,
           zorder=5, label='Training points')

ax.set_xlabel("Time (minutes)", fontsize=12)
ax.set_ylabel("Temperature (°C)", fontsize=12)
ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
ax.set_xlim(df["time"].min() - 20, df["time"].max() + 20)
ax.grid(True, alpha=0.15)
ax.tick_params(labelsize=11)

fig.tight_layout()
fig.savefig(RESULTS_DIR / "gp_posterior_rbf.png",
            dpi=200, bbox_inches='tight')
print("Saved: gp_posterior_rbf.png")
plt.show()

# ============================================================
# Kernel comparison — 5 points, sklearn, fair bounds
# ============================================================

np.random.seed(42)

n_train = 5
X_k = df[["time", "temperature"]].to_numpy()
X_k_norm = (X_k - X_k.mean(axis=0)) / X_k.std(axis=0)
kmeans = KMeans(n_clusters=n_train, random_state=42, n_init=10).fit(X_k_norm)

selected_indices = []
for c_norm in kmeans.cluster_centers_:
    dists = np.linalg.norm(X_k_norm - c_norm, axis=1)
    selected_indices.append(int(np.argmin(dists)))
selected_indices = np.sort(np.unique(selected_indices))
if len(selected_indices) < n_train:
    selected_indices = np.linspace(0, len(df) - 1, n_train, dtype=int)

X_train = df["time"].iloc[selected_indices].to_numpy().reshape(-1, 1)
y_train = df["temperature"].iloc[selected_indices].to_numpy().reshape(-1, 1)
X_all = df["time"].to_numpy().reshape(-1, 1)

print(f"Training points ({len(selected_indices)}):")
for i in range(len(X_train)):
    print(f"  t = {X_train[i,0]:.1f} min, T = {y_train[i,0]:.2f} °C")

# --- Scale on full data ---
scaler_X = StandardScaler().fit(X_all)
scaler_y = StandardScaler().fit(df["temperature"].to_numpy().reshape(-1, 1))

X_train_s = scaler_X.transform(X_train)
y_train_s = scaler_y.transform(y_train).ravel()
X_all_s = scaler_X.transform(X_all)

def unscale(mean_s, std_s=None):
    mean_o = scaler_y.inverse_transform(mean_s.reshape(-1, 1)).flatten()
    if std_s is None:
        return mean_o, None, None
    lower_o = scaler_y.inverse_transform((mean_s - 1.96 * std_s).reshape(-1, 1)).flatten()
    upper_o = scaler_y.inverse_transform((mean_s + 1.96 * std_s).reshape(-1, 1)).flatten()
    return mean_o, lower_o, upper_o

# --- Kernels with IDENTICAL bounds ---
kernel_configs = {
    r"Squared Exponential": C(1.0, (1e-3, 1e3)) * RBF(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e4)),
    r"Matérn $\nu=1/2$": C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e4), nu=0.5),
    r"Matérn $\nu=3/2$": C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e4), nu=1.5),
    r"Matérn $\nu=5/2$": C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e4), nu=2.5),
}

# --- Fit ---
results = {}
for name, kernel in kernel_configs.items():
    gpr = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=50,
        normalize_y=False, random_state=0
    )
    gpr.fit(X_train_s, y_train_s)
    mean_s, std_s = gpr.predict(X_all_s, return_std=True)
    mean_o, lower_o, upper_o = unscale(mean_s, std_s)
    lml = gpr.log_marginal_likelihood_value_
    results[name] = (mean_o, lower_o, upper_o, lml)
    print(f"{name}: LML = {lml:.3f}, kernel = {gpr.kernel_}")

# --- 4-panel figure ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
panel_labels = ["(a)", "(b)", "(c)", "(d)"]

for ax, (name, (mean, lower, upper, lml)), label in zip(
        axes.flat, results.items(), panel_labels):
    ax.plot(df["time"], df["temperature"], color='grey', linewidth=1.0,
            alpha=0.7, label="Simulated data")
    ax.fill_between(df["time"].values, lower, upper,
                    alpha=0.20, color='#4A90D9', label=r"95% credible band")
    ax.plot(df["time"], mean, color='#2C5F8A', linewidth=1.8,
            label="GP mean")
    ax.scatter(X_train.flatten(), y_train.flatten(), marker='o', s=60,
               facecolors='black', edgecolors='white', linewidths=1.0,
               zorder=5, label='Training points')
    ax.set_title(f"{label} {name}", fontsize=12)
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)

axes[1, 0].set_xlabel("Time (minutes)", fontsize=11)
axes[1, 1].set_xlabel("Time (minutes)", fontsize=11)
axes[0, 0].set_ylabel("Temperature (°C)", fontsize=11)
axes[1, 0].set_ylabel("Temperature (°C)", fontsize=11)

fig.tight_layout()
fig.savefig(RESULTS_DIR / "kernel_comparision_4panel.png",
            dpi=200, bbox_inches='tight')
print("Saved: kernel_comparison_4panel.png")
plt.show()

# --- Mean comparison overlay with confidence bands ---
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(df["time"], df["temperature"], color='grey', linewidth=1.0,
        alpha=0.7, linestyle='-', label="Simulated data")

colors = ['#E74C3C', '#E67E22', '#2ECC71', '#2C5F8A']
for (name, (mean, lower, upper, _)), c in zip(results.items(), colors):
    ax.fill_between(df["time"].values, lower, upper,
                    alpha=0.10, color=c)
    ax.plot(df["time"], mean, color=c, linewidth=1.8, label=name)

ax.scatter(X_train.flatten(), y_train.flatten(), marker='o', s=70,
           facecolors='black', edgecolors='white', linewidths=1.0,
           zorder=5, label='Training points')

ax.set_xlabel("Time (minutes)", fontsize=12)
ax.set_ylabel("Temperature (°C)", fontsize=12)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.15)
ax.tick_params(labelsize=11)

fig.tight_layout()
fig.savefig(RESULTS_DIR / "mean_comparison.png",
            dpi=200, bbox_inches='tight')
print("Saved: mean_comparison.png")
plt.show()
