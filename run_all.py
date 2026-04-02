# run_all.py

from robust_ggm_model import GGMConfig, generate_ar2_precision
from frobenius_comparison import compare_gamma_levels, generate_ar1_precision
from uncertainty_metrics import run_uncertainty_comparison

# =============================
# SETTINGS (match your paper)
# =============================
dimensions = [12, 50]

# repetitions
n_reps_frobenius = {12: 100, 50: 100}
n_reps_uncertainty = 30

# =============================
# MAIN LOOP
# =============================
for p in dimensions:

    print("\n" + "=" * 80)
    print(f"RUNNING SIMULATIONS FOR p = {p}")
    print("=" * 80)

    config = GGMConfig(
        p=p,
        n=200,
        gamma=0.01,
        tau_scale=1.0,
        lambda_scale=1.0,
        diag_rate=0.5,
        max_iters=800,
        patience=50,
        adam_lr=0.001,
        seed=42,
        c2_min=0.05,
        c2_max=5.0,
        c2_init=0.5,
        B_samples=100
    )

    # =============================
    # (A) FROBENIUS ERROR
    # =============================
    print("\n--- FROBENIUS ERROR ---")

    # AR1
    print("\nAR1")
    Omega_true = generate_ar1_precision(p)
    compare_gamma_levels(
        Omega_true,
        config,
        n_reps=n_reps_frobenius[p],
        graph_name=f"AR1_p{p}"
    )

    # AR2
    print("\nAR2")
    Omega_true = generate_ar2_precision(p)
    compare_gamma_levels(
        Omega_true,
        config,
        n_reps=n_reps_frobenius[p],
        graph_name=f"AR2_p{p}"
    )

    # =============================
    # (B) UNCERTAINTY METRICS
    # =============================
    print("\n--- UNCERTAINTY METRICS (RMSE, CP, AL) ---")

    # AR1
    print("\nAR1")
    Omega_true = generate_ar1_precision(p)
    run_uncertainty_comparison(
        Omega_true,
        config,
        n_reps=n_reps_uncertainty,
        graph_name=f"AR1_p{p}"
    )

    # AR2
    print("\nAR2")
    Omega_true = generate_ar2_precision(p)
    run_uncertainty_comparison(
        Omega_true,
        config,
        n_reps=n_reps_uncertainty,
        graph_name=f"AR2_p{p}"
    )

print("\n" + "=" * 80)
print("ALL SIMULATIONS COMPLETED")
print("=" * 80)
