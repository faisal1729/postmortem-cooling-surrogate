import os
import numpy as np
import gpflow
from scipy.optimize import curve_fit
from gpflow.optimizers import Scipy
from kaskadeio import runkaskade
from config import *


# ── Marshall-Hoare model ──────────────────────────────────────────────────────

def marshall_hoare(t, A, B, T0, Ta):
    """
    Marshall-Hoare double-exponential model for post-mortem cooling.
    Handles the removable singularity at A = 1 via L'Hopital regularization.

    Parameters
    ----------
    t  : array-like, time since death (hours)
    A  : float, shape parameter (typically A > 1)
    B  : float, decay rate (typically B < 0)
    T0 : float, initial body temperature (°C)
    Ta : float, ambient temperature (°C)

    Returns
    -------
    T : array-like, rectal temperature at time t (°C)
    """
    r1         = np.clip(B * t, -709, 709)
    delta      = A - 1.0
    near_one   = np.abs(delta) < 1e-6
    delta_safe = np.where(near_one, 1.0, delta)
    x          = np.clip((B * t) / delta_safe, -709, 709)
    second     = np.where(near_one, 0.0, (1.0 - A) * np.exp(x))
    theta      = np.exp(r1) * (A + second)
    return Ta + (T0 - Ta) * theta


# ── MH fitting ────────────────────────────────────────────────────────────────

def fit_mh_curve(t_data, T_data, T0=37.0, Ta=21.0,
                 A_guess=1.2815, B_guess=-0.114):
    """
    Fit the Marshall-Hoare model to a single cooling curve via NLS.

    Parameters
    ----------
    t_data  : array, time points in hours
    T_data  : array, temperatures in °C
    T0      : float, initial body temperature (°C)
    Ta      : float, ambient temperature (°C)
    A_guess : float, initial guess for A
    B_guess : float, initial guess for B

    Returns
    -------
    A_opt, B_opt : fitted parameter values
    A_var, B_var : marginal variances from the covariance matrix
    """
    # Exclude t = 0 and T at or below ambient
    valid = (T_data > Ta) & (t_data > 0)
    t_fit = t_data[valid]
    T_fit = T_data[valid]

    if len(t_fit) < 2:
        return np.nan, np.nan, np.nan, np.nan

    try:
        popt, pcov = curve_fit(
            lambda t, A, B: marshall_hoare(t, A, B, T0=T0, Ta=Ta),
            t_fit,
            T_fit,
            p0=(A_guess, B_guess),
            method='lm',
            maxfev=5000
        )
        A_opt, B_opt = popt
        A_var, B_var = pcov[0, 0], pcov[1, 1]
        return A_opt, B_opt, A_var, B_var

    except Exception as e:
        print(f"  MH fitting failed: {e}")
        return np.nan, np.nan, np.nan, np.nan


# ── .gnu file parser ──────────────────────────────────────────────────────────

def parse_gnu_file(filepath):
    """
    Parse time-temperature pairs from a Kaskade .gnu output file.
    Physical parameters are not extracted here — they are already
    known from the simulation configuration that generated the file.
    Header parsing for physical parameters is handled separately
    in analysis.py for Google Colab.

    Parameters
    ----------
    filepath : str, full path to .gnu file

    Returns
    -------
    t : array of time values in hours
    T : array of temperature values in °C
    """
    t_list = []
    T_list = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 2:
                try:
                    t_list.append(float(parts[0]))
                    T_list.append(float(parts[1]))
                except ValueError:
                    continue

    t = np.array(t_list) / 60.0  # convert minutes to hours
    T = np.array(T_list)

    return t, T


# ── Simulation runner ─────────────────────────────────────────────────────────

def run_simulation(sim_index, param_dict):
    """
    Run a single Kaskade FE simulation and return fitted MH parameters.

    Parameters
    ----------
    sim_index  : int, simulation index (1-120)
    param_dict : dict with keys matching PARAM_COLS

    Returns
    -------
    result : dict with physical params + sim_index + A, B, A_var, B_var
             returns None if simulation or fitting failed
    """
    kaskade_params = {
        '--hCapM'     : param_dict['hCapM'],
        '--hConM'     : param_dict['hConM'],
        '--densityM'  : param_dict['densityM'],
        '--convection': param_dict['convection'],
        '--height'    : param_dict['height'],
        '--index'     : sim_index
    }

    print(f"  Running simulation {sim_index} ...")
    runkaskade(RUNPATH, EXE, kaskade_params)

    # Locate output file
    gnu_file = os.path.join(DATA_DIR, f"coolingCurve{sim_index}.gnu")

    if not os.path.exists(gnu_file):
        print(f"  Warning: output file not found for simulation {sim_index}")
        return None

    # Parse time-temperature pairs
    t, T = parse_gnu_file(gnu_file)

    # Fit MH model
    A_opt, B_opt, A_var, B_var = fit_mh_curve(t, T, T0=T0, Ta=TA)

    if np.isnan(A_opt):
        print(f"  Warning: MH fitting failed for simulation {sim_index}")
        return None

    result = {
        'sim_index' : sim_index,
        **param_dict,
        'A'         : A_opt,
        'B'         : B_opt,
        'A_var'     : A_var,
        'B_var'     : B_var
    }

    return result


# ── GP model builder ──────────────────────────────────────────────────────────

def build_gp_model(X, Y):
    """
    Build a GPflow GP regression model with a Matern-5/2 ARD kernel.
    One length scale per input dimension allows the model to learn
    which physical parameters A and B are most sensitive to.

    Parameters
    ----------
    X : array, shape (n, d) — scaled input features
    Y : array, shape (n, 1) — output values (A or B)

    Returns
    -------
    model : gpflow.models.GPR
    """
    kernel = gpflow.kernels.Matern52(
        lengthscales=np.ones(X.shape[1]),
        variance=1.0
    )
    model = gpflow.models.GPR(
        data=(X.astype(np.float64), Y.astype(np.float64)),
        kernel=kernel,
        noise_variance=NOISE_VARIANCE
    )
    return model


# ── GP optimizer ──────────────────────────────────────────────────────────────

def optimize_gp(model):
    """
    Optimize GP hyperparameters by maximizing the log marginal likelihood
    using L-BFGS-B via GPflow's Scipy wrapper.

    Parameters
    ----------
    model : gpflow.models.GPR

    Returns
    -------
    result : scipy optimization result
    """
    opt    = Scipy()
    result = opt.minimize(
        model.training_loss,
        model.trainable_variables,
        options=dict(maxiter=GP_MAXITER)
    )
    return result


# ── Acquisition function ──────────────────────────────────────────────────────

def compute_acquisition(gp_A, gp_B, X_candidates):
    """
    Compute pure GP variance acquisition function over candidate points.
    Acquisition score = predictive variance of A + predictive variance of B.
    Points with high scores are most uncertain and most valuable to simulate.

    Parameters
    ----------
    gp_A         : trained GPR model for A
    gp_B         : trained GPR model for B
    X_candidates : array, shape (n_candidates, d) — scaled candidate points

    Returns
    -------
    scores : array, shape (n_candidates,) — acquisition score per candidate
    """
    _, var_A = gp_A.predict_f(X_candidates.astype(np.float64))
    _, var_B = gp_B.predict_f(X_candidates.astype(np.float64))

    scores = var_A.numpy().flatten() + var_B.numpy().flatten()
    return scores


# ── Prediction error evaluator ────────────────────────────────────────────────

def compute_prediction_errors(gp_A, gp_B, X_test_scaled,
                               A_test, B_test, t_grid):
    """
    Compute RMSE on A, B, and reconstructed cooling curves
    for a fixed test set. Used at each adaptive iteration to
    track how prediction accuracy improves with more simulations.

    Parameters
    ----------
    gp_A          : trained GPR model for A
    gp_B          : trained GPR model for B
    X_test_scaled : array, shape (n_test, d) — scaled test inputs
    A_test        : array, shape (n_test,)   — true A values
    B_test        : array, shape (n_test,)   — true B values
    t_grid        : array — time grid for curve reconstruction (hours)

    Returns
    -------
    rmse_A     : float — RMSE on parameter A
    rmse_B     : float — RMSE on parameter B
    curve_rmse : float — mean curve reconstruction RMSE across test points (°C)
    """
    A_pred = gp_A.predict_f(
        X_test_scaled.astype(np.float64))[0].numpy().flatten()
    B_pred = gp_B.predict_f(
        X_test_scaled.astype(np.float64))[0].numpy().flatten()

    rmse_A = np.sqrt(np.mean((A_test - A_pred) ** 2))
    rmse_B = np.sqrt(np.mean((B_test - B_pred) ** 2))

    # Curve reconstruction RMSE
    curve_rmses = []
    for j in range(len(A_test)):
        T_true = marshall_hoare(t_grid, A_test[j], B_test[j], T0, TA)
        T_pred = marshall_hoare(t_grid, A_pred[j], B_pred[j], T0, TA)
        curve_rmses.append(np.sqrt(np.mean((T_true - T_pred) ** 2)))

    curve_rmse = np.mean(curve_rmses)

    return rmse_A, rmse_B, curve_rmse