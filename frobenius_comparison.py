"""
===========================================================================
COMPARISON OF ROBUSTNESS LEVELS (γ = 0.01 vs γ = 0.02) - FROBENIUS ERROR ONLY
===========================================================================
Compares Laplace vs Horseshoe priors with γ = 0.01 and γ = 0.02
for AR(1) and AR(2) graph structures across multiple dimensions.

Metrics: Normalized Frobenius error only (precision matrix estimation accuracy)
"""

import numpy as np
import pickle
import time
import pandas as pd
from scipy import stats
import torch
from torch.optim import Adam

# Import from main code
from robust_ggm_model import (
    GGMConfig, tri_idx_torch, gamma_divergence_loss,
    build_precision_matrix, partial_corr_from_Omega,
    generate_ar2_precision, add_onizuka_contamination,
    fit_map_data_adaptive, LARGE_NONPD_LOSS
)

# ===========================================================================
# AR(1) PRECISION GENERATOR
# ===========================================================================

def generate_ar1_precision(p, rho=0.5):
    """Generate AR(1) precision matrix."""
    Omega = np.eye(p)
    for i in range(1, p):
        Omega[i, i-1] = -rho
        Omega[i-1, i] = -rho
    Omega += 0.1 * np.eye(p)
    return Omega

# ===========================================================================
# EVALUATION FUNCTION - NORMALIZED FROBENIUS ERROR ONLY
# ===========================================================================

def evaluate_frobenius(Omega_hat, Omega_true):
    """
    Compute normalized Frobenius error for precision matrix estimation.
    
    Parameters
    ----------
    Omega_hat : array
        Estimated precision matrix
    Omega_true : array
        True precision matrix
        
    Returns
    -------
    frob_error : float
        Normalized Frobenius error ||Ω̂ - Ω||_F / ||Ω||_F
    """
    return np.linalg.norm(Omega_hat - Omega_true, 'fro') / (np.linalg.norm(Omega_true, 'fro') + 1e-10)

# ===========================================================================
# HEAVY-TAILED CONTAMINATION FUNCTION
# ===========================================================================

def add_heavytail_contamination(Y, epsilon=0.1, df=3):
    """Add heavy-tailed contamination using t-distribution."""
    n, p = Y.shape
    n_outliers = int(n * epsilon)
    
    if n_outliers == 0:
        return Y.copy()
    
    Y_contam = Y.copy()
    outlier_indices = np.random.choice(n, n_outliers, replace=False)
    
    outliers = stats.t.rvs(df=df, size=(n_outliers, p))
    outliers = outliers * np.std(Y, axis=0) + np.mean(Y, axis=0)
    Y_contam[outlier_indices] = outliers
    
    return Y_contam

# ===========================================================================
# LAPLACE PRIOR WITH FIXED SHRINKAGE
# ===========================================================================

def laplace_prior_negative_log(beta, lambda_global):
    """Laplace prior negative log density (Lasso penalty)."""
    return lambda_global * torch.sum(torch.abs(beta))

def total_loss_laplace_prior(Y, tri_vec, diag_uncon, config, weights=None):
    """Total loss with Laplace prior (Onizuka's method)."""
    p = config.p
    device = config.device
    
    if weights is None:
        weights = torch.ones(Y.shape[0], device=device, dtype=torch.float64) / Y.shape[0]
    
    rows, cols = tri_idx_torch(p, device)
    L = torch.zeros((p, p), dtype=torch.float64, device=device)
    L[rows, cols] = tri_vec
    L_diag = torch.exp(torch.clamp(diag_uncon, -8.0, 8.0))
    diag_idx = torch.arange(p, device=device)
    L[diag_idx, diag_idx] = L_diag
    Omega = L @ L.T
    Omega = 0.5 * (Omega + Omega.T)
    
    lik, ok = gamma_divergence_loss(Omega, Y, weights, config.gamma)
    if not ok:
        return torch.tensor(LARGE_NONPD_LOSS, device=device), False, {}
    
    lambda_global = np.sqrt(np.log(config.p) / config.n)
    prior = laplace_prior_negative_log(tri_vec, lambda_global)
    diag_prior = config.diag_rate * torch.sum(L_diag) - torch.sum(diag_uncon)
    
    total_loss = lik + prior + diag_prior
    
    details = {"Omega": Omega.detach()}
    return total_loss, True, details

def fit_map_laplace(Y, config, weights=None, verbose=False):
    """MAP estimation with Laplace prior."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    n, p = Y.shape
    device = config.device
    
    Y_tensor = torch.tensor(Y, dtype=torch.float64, device=device)
    
    if weights is None:
        weights_arr = np.ones(n) / n
    else:
        weights_arr = np.asarray(weights, dtype=float)
        weights_arr = weights_arr / (weights_arr.sum() + 1e-30)
    
    weights_tensor = torch.tensor(weights_arr, dtype=torch.float64, device=device)
    
    mean_w = (weights_arr[:, None] * Y).sum(axis=0)
    Yc = Y - mean_w
    Yw = Yc * np.sqrt(weights_arr)[:, None]
    S = Yw.T @ Yw
    S = 0.5 * (S + S.T)
    
    ridge_coeff = max(1e-6, 1e-3 * np.trace(S) / p)
    prec_init = np.linalg.inv(S + ridge_coeff * np.eye(p))
    prec_init = 0.5 * (prec_init + prec_init.T)
    
    try:
        L_init = np.linalg.cholesky(prec_init)
    except np.linalg.LinAlgError:
        L_init = np.eye(p) * np.sqrt(np.clip(np.diag(prec_init), 1e-6, None))
    
    rows, cols = tri_idx_torch(p, device)
    tri_init = np.array([L_init[i, j] for i in range(1, p) for j in range(i)])
    diag_init = np.log(np.clip(np.diag(L_init), 1e-6, None))
    
    tri_vec = torch.tensor(tri_init, device=device, requires_grad=True)
    diag_uncon = torch.tensor(diag_init, device=device, requires_grad=True)
    
    params = [tri_vec, diag_uncon]
    optimizer = Adam(params, lr=config.adam_lr, eps=1e-8)
    
    best_loss = float("inf")
    best_params = None
    patience_counter = 0
    
    for it in range(config.max_iters):
        optimizer.zero_grad()
        loss, ok, details = total_loss_laplace_prior(Y_tensor, tri_vec, diag_uncon, config, weights_tensor)
        
        if not ok:
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        
        loss_val = float(loss.item())
        
        if loss_val < best_loss - 1e-6:
            best_loss = loss_val
            best_params = {
                "tri_vec": tri_vec.detach().clone(),
                "diag_uncon": diag_uncon.detach().clone()
            }
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter > config.patience:
            break
    
    if best_params is not None:
        with torch.no_grad():
            tri_vec.copy_(best_params["tri_vec"])
            diag_uncon.copy_(best_params["diag_uncon"])
    
    with torch.no_grad():
        L = torch.zeros((p, p), dtype=torch.float64, device=device)
        L[rows, cols] = tri_vec
        diag_idx = torch.arange(p, device=device)
        L[diag_idx, diag_idx] = torch.exp(torch.clamp(diag_uncon, -8.0, 8.0))
        Omega_hat = (L @ L.T).cpu().numpy()
    
    return {
        "Omega_hat": Omega_hat,
        "success": best_params is not None
    }

# ===========================================================================
# DATA GENERATION
# ===========================================================================

def generate_data_for_scenario(Omega_true, config, rep, contamination_type="a", epsilon=0.1, eta=10):
    """Generate data with specified contamination."""
    type_hash = abs(hash(contamination_type)) % 500
    np.random.seed(1000 * rep + type_hash)
    
    Sigma = np.linalg.inv(Omega_true)
    Y = np.random.multivariate_normal(np.zeros(config.p), Sigma, config.n)
    Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-8)
    
    if contamination_type == "a":
        return Y
    elif contamination_type in ["b", "c"]:
        Y_contam, _ = add_onizuka_contamination(Y, Omega_true, contam_type=contamination_type, 
                                                epsilon=epsilon, eta=eta)
        return Y_contam
    elif contamination_type == "ht":
        return add_heavytail_contamination(Y, epsilon=epsilon, df=3)
    else:
        raise ValueError(f"Unknown contamination type: {contamination_type}")

# ===========================================================================
# COMPARISON FUNCTION - FROBENIUS ERROR ONLY
# ===========================================================================

def compare_gamma_levels(Omega_true, base_config, n_reps, graph_name="AR2"):
    """
    Compare Laplace and Horseshoe priors with γ = 0.01 and 0.02.
    Returns only normalized Frobenius error.
    """
    scenarios = [
        {"name": "Clean", "type": "a", "epsilon": 0.0, "eta": 0},
        {"name": "Var_contam", "type": "b", "epsilon": 0.1, "eta": 0},
        {"name": "Mean_eta10", "type": "c", "epsilon": 0.1, "eta": 10},
        {"name": "HeavyTail_df3", "type": "ht", "epsilon": 0.15, "eta": 0}
    ]
    
    gamma_values = [0.01, 0.02]
    
    method_keys = {
        ("laplace", 0.01): "Laplace_γ001",
        ("laplace", 0.02): "Laplace_γ002",
        ("horseshoe", 0.01): "Horseshoe_γ001",
        ("horseshoe", 0.02): "Horseshoe_γ002"
    }
    
    results = {key: {s["name"]: [] for s in scenarios} for key in method_keys.values()}
    
    p = base_config.p
    start_time = time.time()
    
    print(f"\n  {graph_name} (p={p}, reps={n_reps})", end="")
    
    for rep in range(n_reps):
        if n_reps > 10 and (rep + 1) % 10 == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"\n    Rep {rep+1}/{n_reps}, {elapsed:.1f} min elapsed", end="")
        
        for scenario in scenarios:
            Y = generate_data_for_scenario(Omega_true, base_config, rep + 1000,
                                          contamination_type=scenario["type"],
                                          epsilon=scenario["epsilon"],
                                          eta=scenario["eta"])
            
            for gamma in gamma_values:
                for prior in ["laplace", "horseshoe"]:
                    method = method_keys[(prior, gamma)]
                    config = GGMConfig(**base_config.__dict__)
                    config.gamma = gamma
                    config.seed = rep + 2000 * scenarios.index(scenario) + \
                                  (0 if prior == "laplace" else 10000) + int(gamma * 1000)
                    
                    if prior == "laplace":
                        fit_result = fit_map_laplace(Y, config, verbose=False)
                    else:
                        fit_result = fit_map_data_adaptive(Y, config, verbose=False)
                    
                    if fit_result["success"]:
                        frob_error = evaluate_frobenius(fit_result["Omega_hat"], Omega_true)
                        results[method][scenario["name"]].append(frob_error)
    
    # Print summary table
    print(f"\n\n  Results for {graph_name} (p={base_config.p}):")
    print(f"  {'Scenario':<18} ", end="")
    for method in method_keys.values():
        print(f"{method:<18} ", end="")
    print()
    print("  " + "-" * (18 + 18 * len(method_keys)))
    
    for scenario in scenarios:
        print(f"  {scenario['name']:<18} ", end="")
        for method in method_keys.values():
            data = results[method][scenario["name"]]
            if data:
                mean_val = np.mean(data)
                std_val = np.std(data)
                print(f"{mean_val:.4f}±{std_val:.4f} ", end="")
            else:
                print(f"{'N/A':<18} ", end="")
        print()
    
    return results

# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

if __name__ == "__main__":
    print("="*80)
    print("COMPARISON OF ROBUSTNESS LEVELS (γ = 0.01 vs 0.02)")
    print("NORMALIZED FROBENIUS ERROR ONLY")
    print("="*80)
    
    base_config = GGMConfig(
        p=12,
        n=200,
        gamma=0.02,
        tau_scale=1.0,
        lambda_scale=1.0,
        diag_rate=0.5,
        max_iters=800,
        patience=50,
        adam_lr=0.001,
        seed=42,
        c2_min=0.05,
        c2_max=5.0,
        c2_init=0.5
    )
    
    # Define dimensions and repetition counts
    dimension_configs = [
        (12, 100, "AR1"),
        (12, 100, "AR2"),
        (50, 100, "AR1"),
        (50, 100, "AR2"),
        (100, 10, "AR1"),
        (100, 10, "AR2")
    ]
    
    all_results = {}
    
    for p, n_reps, graph_type in dimension_configs:
        print(f"\n{'='*80}")
        print(f"RUNNING: p={p}, {graph_type} graph, {n_reps} repetitions")
        print(f"{'='*80}")
        
        config = GGMConfig(**base_config.__dict__)
        config.p = p
        
        if graph_type == "AR1":
            Omega_true = generate_ar1_precision(p)
        else:
            Omega_true = generate_ar2_precision(p)
        
        results = compare_gamma_levels(Omega_true, config, n_reps, graph_name=f"{graph_type}_p{p}")
        all_results[f"{graph_type}_p{p}"] = results
    
    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")
    
    with open("frobenius_comparison_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    
    # Create comprehensive summary table
    summary_rows = []
    for key, results_dict in all_results.items():
        for method, scenarios in results_dict.items():
            for scenario, data in scenarios.items():
                if data:
                    summary_rows.append({
                        "Setting": key,
                        "Method": method,
                        "Scenario": scenario,
                        "Frobenius_Mean": np.mean(data),
                        "Frobenius_SD": np.std(data)
                    })
    
    df_summary = pd.DataFrame(summary_rows)
    
    # Pivot for cleaner display
    pivot = df_summary.pivot_table(
        values='Frobenius_Mean',
        index=['Setting', 'Scenario'],
        columns='Method',
        aggfunc='first'
    ).round(4)
    
    print("\nFinal Summary Table - Normalized Frobenius Error:")
    print("(Lower values indicate better precision matrix estimation)")
    print("\n", pivot)
    
    # Save CSV
    df_summary.to_csv("frobenius_comparison_summary.csv", index=False)
    pivot.to_csv("frobenius_comparison_pivot.csv")
    
    print(f"\n✅ Results saved to:")
    print(f"   - frobenius_comparison_results.pkl (full results)")
    print(f"   - frobenius_comparison_summary.csv (long format)")
    print(f"   - frobenius_comparison_pivot.csv (pivot table)")
