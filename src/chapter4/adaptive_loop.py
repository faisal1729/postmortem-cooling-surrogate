import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler
from config import *
from core_functions import *

# ── Part 1: Initial Setup ─────────────────────────────────────────────────────

if os.path.exists(CSV_TRAINING) and os.path.exists(CSV_TEST):
    print("=" * 55)
    print("  Phase 1 already complete — skipping simulations")
    print(f"  Found: {CSV_TRAINING}")
    print(f"  Found: {CSV_TEST}")
    print("=" * 55)

else:
    print("=" * 55)
    print("  Generating test simulations (indices 101-120)")
    print("=" * 55)

    sampler_test = qmc.Sobol(d=len(PARAM_COLS), scramble=True,
                              seed=RANDOM_SEED + 1)
    samples_test = sampler_test.random(N_TEST)

    l_bounds = [PARAM_RANGES_TEST[k][0] for k in PARAM_COLS]
    u_bounds = [PARAM_RANGES_TEST[k][1] for k in PARAM_COLS]
    scaled_test = qmc.scale(samples_test, l_bounds, u_bounds)

    test_params_df = pd.DataFrame(scaled_test, columns=PARAM_COLS)
    test_results   = []

    for i, row in test_params_df.iterrows():
        sim_index = 101 + i
        result    = run_simulation(sim_index, row.to_dict())
        if result is not None:
            test_results.append(result)

    test_df = pd.DataFrame(test_results)
    test_df.to_csv(CSV_TEST, index=False)

    print(f"\nTest simulations complete: {len(test_df)} / {N_TEST}")
    print(f"Saved to: {CSV_TEST}")

    # ── Initial Sobol training simulations ────────────────────────────────
    print("\n" + "=" * 55)
    print("  Running initial Sobol simulations (indices 1-20)")
    print("=" * 55)

    sampler_init = qmc.Sobol(d=len(PARAM_COLS), scramble=True,
                              seed=RANDOM_SEED)
    samples_init = sampler_init.random(N_INITIAL)

    l_bounds = [PARAM_RANGES_TRAIN[k][0] for k in PARAM_COLS]
    u_bounds = [PARAM_RANGES_TRAIN[k][1] for k in PARAM_COLS]
    scaled_init = qmc.scale(samples_init, l_bounds, u_bounds)

    init_params_df   = pd.DataFrame(scaled_init, columns=PARAM_COLS)
    training_results = []

    for i, row in init_params_df.iterrows():
        sim_index = 1 + i
        result    = run_simulation(sim_index, row.to_dict())
        if result is not None:
            result['iteration'] = 0
            training_results.append(result)

    training_df = pd.DataFrame(training_results)
    training_df.to_csv(CSV_TRAINING, index=False)

    print(f"\nInitial simulations complete: {len(training_df)} / {N_INITIAL}")
    print(f"Saved to: {CSV_TRAINING}")


# ── Part 2: Adaptive Loop ─────────────────────────────────────────────────────

T_GRID = np.linspace(0, 20, 241)  # 0 to 20 hours in 5-minute intervals (241 points)
print("\n" + "=" * 55)
print("  Starting Adaptive Loop")
print("=" * 55)

# ── Load existing data ────────────────────────────────────────────────────────

training_df = pd.read_csv(CSV_TRAINING)
test_df     = pd.read_csv(CSV_TEST)

print(f"\nLoaded training data : {len(training_df)} points")
print(f"Loaded test data     : {len(test_df)} points")

# ── Prepare test set — fixed throughout all iterations ───────────────────────

X_test  = test_df[PARAM_COLS].values.astype(np.float64)
A_test  = test_df['A'].values.astype(np.float64)
B_test  = test_df['B'].values.astype(np.float64)

# ── Generate candidate grid — fixed throughout all iterations ─────────────────
# A dense random grid over the training parameter space.
# The acquisition function is evaluated over this grid at every iteration
# to find the most uncertain location for the next simulation.

rng         = np.random.default_rng(RANDOM_SEED)
l_bounds    = np.array([PARAM_RANGES_TRAIN[k][0] for k in PARAM_COLS])
u_bounds    = np.array([PARAM_RANGES_TRAIN[k][1] for k in PARAM_COLS])
X_candidates_raw = rng.uniform(
    low=l_bounds, high=u_bounds,
    size=(N_CANDIDATES, len(PARAM_COLS))
)

# ── Initialize metrics storage ────────────────────────────────────────────────

metrics_records = []

# ── Determine starting iteration ─────────────────────────────────────────────
# If the loop was interrupted and restarted, we resume from where we left off
# rather than rerunning completed iterations.

if os.path.exists(CSV_METRICS):
    metrics_df      = pd.read_csv(CSV_METRICS)
    completed_iters = len(metrics_df)
    print(f"\nResuming from iteration {completed_iters + 1}")
else:
    completed_iters = 0
    print("\nStarting fresh adaptive loop from iteration 1")

# ── Adaptive loop ─────────────────────────────────────────────────────────────

for iteration in range(1, N_ADAPTIVE + 1):

    # Skip already completed iterations if resuming
    if iteration <= completed_iters:
        continue

    sim_index = N_INITIAL + iteration   # indices 21 to 100
    n_current = len(training_df)

    print(f"\n{'─' * 55}")
    print(f"  Iteration {iteration} / {N_ADAPTIVE} "
          f"— simulation index {sim_index} "
          f"— training points: {n_current}")
    print(f"{'─' * 55}")

    # ── Scale inputs ──────────────────────────────────────────────────────
    # Fit scaler on current training data only
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(
        training_df[PARAM_COLS].values.astype(np.float64)
    )
    Y_A_train = training_df['A'].values.reshape(-1, 1).astype(np.float64)
    Y_B_train = training_df['B'].values.reshape(-1, 1).astype(np.float64)

    # Scale test set and candidate grid using same scaler
    X_test_scaled       = scaler.transform(X_test)
    X_candidates_scaled = scaler.transform(X_candidates_raw)

    # ── Build and optimize GP models ──────────────────────────────────────
    print("  Training GP models ...")
    gp_A = build_gp_model(X_train, Y_A_train)
    gp_B = build_gp_model(X_train, Y_B_train)
    optimize_gp(gp_A)
    optimize_gp(gp_B)

    # ── Evaluate acquisition function over candidate grid ─────────────────
    scores = compute_acquisition(gp_A, gp_B, X_candidates_scaled)

    max_acquisition      = float(np.max(scores))
    mean_integ_variance  = float(np.mean(scores))

    print(f"  Max acquisition score    : {max_acquisition:.6f}")
    print(f"  Mean integrated variance : {mean_integ_variance:.6f}")

    # ── Select next simulation point ──────────────────────────────────────
    best_idx    = np.argmax(scores)
    x_next_raw  = X_candidates_raw[best_idx]   # unscaled — for Kaskade
    next_params = dict(zip(PARAM_COLS, x_next_raw))

    print(f"  Selected next point:")
    for k, v in next_params.items():
        print(f"    {k} = {v:.4f}")

    # ── Run FE simulation at selected point ───────────────────────────────
    result = run_simulation(sim_index, next_params)

    if result is None:
        print(f"  Simulation {sim_index} failed — skipping iteration.")
        continue

    result['iteration'] = iteration

    # ── Add to training dataset ───────────────────────────────────────────
    new_row     = pd.DataFrame([result])
    training_df = pd.concat([training_df, new_row], ignore_index=True)
    training_df.to_csv(CSV_TRAINING, index=False)

    print(f"  Fitted A = {result['A']:.4f},  B = {result['B']:.6f}")

    # ── Evaluate prediction errors on fixed test set ──────────────────────
    print("  Evaluating prediction errors on test set ...")
    rmse_A, rmse_B, curve_rmse = compute_prediction_errors(
        gp_A, gp_B,
        X_test_scaled,
        A_test, B_test,
        T_GRID
    )

    print(f"  RMSE A      : {rmse_A:.4f}")
    print(f"  RMSE B      : {rmse_B:.6f}")
    print(f"  Curve RMSE  : {curve_rmse:.4f} °C")

    # ── Save metrics for this iteration ───────────────────────────────────
    metrics_records.append({
        'iteration'             : iteration,
        'n_points'              : n_current + 1,
        'max_acquisition'       : max_acquisition,
        'mean_integrated_variance': mean_integ_variance,
        'rmse_A'                : rmse_A,
        'rmse_B'                : rmse_B,
        'curve_rmse'            : curve_rmse
    })

    # Append to CSV after every iteration — nothing lost if loop crashes
    metrics_df = pd.DataFrame(metrics_records)
    if completed_iters > 0 and os.path.exists(CSV_METRICS):
        existing    = pd.read_csv(CSV_METRICS)
        metrics_df  = pd.concat([existing, metrics_df], ignore_index=True) \
                      if iteration == completed_iters + 1 \
                      else existing
        metrics_df  = pd.read_csv(CSV_METRICS) if iteration > completed_iters + 1 \
                      else pd.concat([pd.read_csv(CSV_METRICS),
                                      pd.DataFrame([metrics_records[-1]])],
                                     ignore_index=True)
    pd.DataFrame(metrics_records if completed_iters == 0
                 else [metrics_records[-1]]).to_csv(
        CSV_METRICS,
        mode='a' if (completed_iters > 0 or iteration > 1) else 'w',
        header=(iteration == 1 and completed_iters == 0),
        index=False
    )

print("\n" + "=" * 55)
print("  Adaptive loop complete")
print(f"  Total training points : {len(training_df)}")
print(f"  Metrics saved to      : {CSV_METRICS}")
print(f"  Training data saved to: {CSV_TRAINING}")
print("=" * 55)