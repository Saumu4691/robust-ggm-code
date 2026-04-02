"""
===========================================================================
UNCERTAINTY COMPARISON: ONIZUKA vs HORSESHOE PRIOR
FOCUS: RMSE, CP, AL (Posterior Uncertainty Metrics)
COMPARING: γ=0.01 vs γ=0.02
STRUCTURES: AR(1) and AR(2)
DIMENSIONS: p = 12 and p = 50
===========================================================================
"""

import numpy as np
import pickle
import time
import pandas as pd
from scipy import stats
import torch
from torch.optim import Adam

# Import from your main code
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
    """Generate AR(1) precision matrix with tridiagonal structure."""
    Omega = np.eye(p)
    for i in range(1, p):
        Omega[i, i-1] = Omega[i-1, i] = -rho
    Omega += 0.1 * np.eye(p)
    return Omega


# ===========================================================================
# UNCERTAINTY METRICS (RMSE, COVERAGE PROBABILITY, AVERAGE LENGTH)
# ===========================================================================

def compute_uncertainty_metrics(Omega_samples, Omega_true, alpha=0.05):
    """
    Compute uncertainty quantification metrics for precision matrix estimation.
    
    Parameters
    ----------
    Omega_samples : array (B, p, p)
        Posterior samples of precision matrix
    Omega_true : array (p, p)
        True precision matrix
    alpha : float
        Significance level for credible intervals (default 0.05 → 95% CI)
    
    Returns
    -------
    rmse : float
        Root Mean Square Error of posterior mean (off-diagonals only)
    al : float
        Average Length of credible intervals (off-diagonals only)
    cp : float
        Coverage Probability (target = 0.95 for well-calibrated uncertainty)
    """
    B, p, _ = Omega_samples.shape
    
    # Focus on off-diagonal elements (the edges)
    mask = np.triu(np.ones((p, p), dtype=bool), k=1)
    
    # Posterior mean
    Omega_mean = np.mean(Omega_samples, axis=0)
    
    # RMSE on off-diagonals
    rmse = np.sqrt(np.mean((Omega_mean[mask] - Omega_true[mask])**2))
    
    # Credible intervals
    lower = np.quantile(Omega_samples, alpha/2, axis=0)
    upper = np.quantile(Omega_samples, 1 - alpha/2, axis=0)
    
    # Average length (off-diagonals only)
    al = np.mean((upper - lower)[mask])
    
    # Coverage probability
    true_offdiag = Omega_true[mask]
    lower_offdiag = lower[mask]
    upper_offdiag = upper[mask]
    
    inside = (true_offdiag >= lower_offdiag) & (true_offdiag <= upper_offdiag)
    cp = np.mean(inside)
    
    return rmse, al, cp


# ===========================================================================
# LAPLACE PRIOR WITH POSTERIOR SAMPLES (ONIZUKA'S METHOD)
# ===========================================================================

def gamma_loss_with_weights(Omega, Y, weights, gamma):
    """γ-divergence loss that returns updated weights."""
    p = Omega.shape[0]
    n = Y.shape[0]
    device = Omega.device
    
    try:
        L = torch.linalg.cholesky(Omega)
        log_det = 2 * torch.sum(torch.log(torch.diag(L)))
        
        Y_Omega = Y @ Omega
        quad = torch.sum(Y_Omega * Y, dim=1)
        
        if gamma > 0:
            c = 1.0 / (1.0 + gamma)
            w = torch.exp(c * (quad - log_det))
            w = w / (w.sum() + 1e-30)
            weights = w.detach()
            lik = -torch.sum(weights * (quad - log_det))
        else:
            lik = 0.5 * torch.sum(quad) - 0.5 * n * log_det
            weights = torch.ones(n, device=device) / n
        
        return lik, True, weights
    except:
        return torch.tensor(LARGE_NONPD_LOSS, device=device), False, None


def fit_laplace_with_uncertainty(Y, config, verbose=False):
    """
    MAP estimation with Laplace prior + bootstrap for uncertainty quantification.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    n, p = Y.shape
    device = config.device
    
    Y_tensor = torch.tensor(Y, dtype=torch.float64, device=device)
    weights = torch.ones(n, device=device, dtype=torch.float64) / n
    
    # Initialize from weighted covariance
    weights_arr = weights.cpu().numpy()
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
    
    lambda_global = np.sqrt(np.log(p) / n)
    
    best_loss = float("inf")
    best_params = None
    patience_counter = 0
    
    for it in range(config.max_iters):
        optimizer.zero_grad()
        
        L = torch.zeros((p, p), dtype=torch.float64, device=device)
        L[rows, cols] = tri_vec
        diag_idx = torch.arange(p, device=device)
        L[diag_idx, diag_idx] = torch.exp(torch.clamp(diag_uncon, -8.0, 8.0))
        Omega = L @ L.T
        Omega = 0.5 * (Omega + Omega.T)
        
        if config.gamma > 0:
            lik, ok, weights = gamma_loss_with_weights(Omega, Y_tensor, weights, config.gamma)
        else:
            lik, ok = gamma_divergence_loss(Omega, Y_tensor, weights, config.gamma)
        
        if not ok:
            continue
        
        prior = lambda_global * torch.sum(torch.abs(tri_vec))
        diag_prior = config.diag_rate * torch.sum(torch.exp(diag_uncon)) - torch.sum(diag_uncon)
        total_loss = lik + prior + diag_prior
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        
        loss_val = float(total_loss.item())
        
        if loss_val < best_loss - 1e-6:
            best_loss = loss_val
            best_params = {"tri_vec": tri_vec.detach().clone(), "diag_uncon": diag_uncon.detach().clone()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter > config.patience:
            break
    
    if best_params is None:
        return {"success": False}
    
    # Bootstrap for uncertainty
    Omega_samples = bootstrap_laplace(Y, config, best_params, rows, cols, lambda_global)
    
    with torch.no_grad():
        L = torch.zeros((p, p), dtype=torch.float64, device=device)
        L[rows, cols] = best_params["tri_vec"]
        diag_idx = torch.arange(p, device=device)
        L[diag_idx, diag_idx] = torch.exp(torch.clamp(best_params["diag_uncon"], -8.0, 8.0))
        Omega_hat = (L @ L.T).cpu().numpy()
        Omega_hat = 0.5 * (Omega_hat + Omega_hat.T)
    
    return {
        "Omega_hat": Omega_hat,
        "Omega_samples": Omega_samples,
        "success": Omega_samples is not None,
        "lambda": lambda_global
    }


def bootstrap_laplace(Y, config, best_params, rows, cols, lambda_global):
    """Generate bootstrap samples for uncertainty quantification."""
    p = config.p
    n = config.n
    samples = []
    
    for b in range(min(config.B_samples, 100)):
        idx = np.random.choice(n, n, replace=True)
        Y_boot = Y[idx]
        
        config_boot = GGMConfig(**config.__dict__)
        config_boot.seed = config.seed + b + 1000
        config_boot.max_iters = 200
        
        result = quick_laplace_map(Y_boot, config_boot, rows, cols, lambda_global)
        if result is not None:
            samples.append(result)
    
    if len(samples) >= 10:
        return np.array(samples)
    else:
        return None


def quick_laplace_map(Y, config, rows, cols, lambda_global):
    """Fast MAP estimation for bootstrap samples."""
    p = config.p
    n = config.n
    device = config.device
    
    Y_tensor = torch.tensor(Y, dtype=torch.float64, device=device)
    weights = torch.ones(n, device=device, dtype=torch.float64) / n
    
    S = np.cov(Y.T)
    S = 0.5 * (S + S.T)
    ridge_coeff = 1e-4
    prec_init = np.linalg.inv(S + ridge_coeff * np.eye(p))
    
    try:
        L_init = np.linalg.cholesky(prec_init)
    except:
        L_init = np.eye(p)
    
    tri_init = np.array([L_init[i, j] for i in range(1, p) for j in range(i)])
    diag_init = np.log(np.clip(np.diag(L_init), 1e-6, None))
    
    tri_vec = torch.tensor(tri_init, device=device, requires_grad=True)
    diag_uncon = torch.tensor(diag_init, device=device, requires_grad=True)
    optimizer = Adam([tri_vec, diag_uncon], lr=0.001)
    
    for _ in range(100):
        optimizer.zero_grad()
        
        L = torch.zeros((p, p), dtype=torch.float64, device=device)
        L[rows, cols] = tri_vec
        diag_idx = torch.arange(p, device=device)
        L[diag_idx, diag_idx] = torch.exp(torch.clamp(diag_uncon, -8.0, 8.0))
        Omega = L @ L.T
        Omega = 0.5 * (Omega + Omega.T)
        
        lik, ok = gamma_divergence_loss(Omega, Y_tensor, weights, config.gamma)
        if not ok:
            return None
        
        prior = lambda_global * torch.sum(torch.abs(tri_vec))
        diag_prior = config.diag_rate * torch.sum(torch.exp(diag_uncon)) - torch.sum(diag_uncon)
        loss = lik + prior + diag_prior
        
        loss.backward()
        optimizer.step()
    
    with torch.no_grad():
        L = torch.zeros((p, p), dtype=torch.float64, device=device)
        L[rows, cols] = tri_vec
        diag_idx = torch.arange(p, device=device)
        L[diag_idx, diag_idx] = torch.exp(torch.clamp(diag_uncon, -8.0, 8.0))
        Omega = (L @ L.T).cpu().numpy()
        Omega = 0.5 * (Omega + Omega.T)
    
    return Omega


# ===========================================================================
# DATA GENERATION WITH CONTAMINATION
# ===========================================================================

def add_heavytail_contamination(Y, epsilon=0.1, df=3):
    """Add heavy-tailed t-distribution contamination."""
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


def generate_contaminated_data(Omega_true, config, rep, contam_type="a", epsilon=0.1, eta=10):
    """Generate clean or contaminated data."""
    type_hash = abs(hash(contam_type)) % 500
    np.random.seed(1000 * rep + type_hash)
    
    Sigma = np.linalg.inv(Omega_true)
    Sigma = 0.5 * (Sigma + Sigma.T)
    
    Y = np.random.multivariate_normal(np.zeros(config.p), Sigma, config.n)
    Y = (Y - Y.mean(axis=0)) / (Y.std(axis=0) + 1e-8)
    
    if contam_type == "a":
        return Y
    elif contam_type in ["b", "c"]:
        Y_contam, _ = add_onizuka_contamination(Y, Omega_true, contam_type=contam_type,
                                                epsilon=epsilon, eta=eta)
        return Y_contam
    elif contam_type == "ht":
        return add_heavytail_contamination(Y, epsilon=epsilon, df=3)
    else:
        raise ValueError(f"Unknown contamination type: {contam_type}")


# ===========================================================================
# MAIN COMPARISON FUNCTION
# ===========================================================================

def run_uncertainty_comparison(Omega_true, config, n_reps=30, graph_name="AR1"):
    """
    Run comprehensive uncertainty comparison for given graph structure.
    Compares Laplace (Onizuka) vs Horseshoe prior with γ=0.01 and γ=0.02.
    """
    scenarios = [
        {"name": "Clean", "type": "a", "epsilon": 0.0, "eta": 0},
        {"name": "Variance", "type": "b", "epsilon": 0.1, "eta": 0},
        {"name": "Mean_shift", "type": "c", "epsilon": 0.1, "eta": 10},
        {"name": "Heavy_tail", "type": "ht", "epsilon": 0.15, "eta": 0}
    ]
    
    methods = {
        "Laplace_γ001": {"prior": "laplace", "gamma": 0.01},
        "Laplace_γ002": {"prior": "laplace", "gamma": 0.02},
        "Horseshoe_γ001": {"prior": "horseshoe", "gamma": 0.01},
        "Horseshoe_γ002": {"prior": "horseshoe", "gamma": 0.02}
    }
    
    results = {method: {s["name"]: [] for s in scenarios} for method in methods.keys()}
    
    p = config.p
    print(f"\n{'='*80}")
    print(f"📊 Running: {graph_name} (p={p}, n={config.n}, reps={n_reps})")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    for rep in range(n_reps):
        if (rep + 1) % 10 == 0 or rep == 0:
            elapsed = (time.time() - start_time) / 60
            print(f"  Repetition {rep+1:3d}/{n_reps} | Elapsed: {elapsed:.1f} min", end="")
        
        for scenario in scenarios:
            Y = generate_contaminated_data(Omega_true, config, rep + 1000,
                                          contam_type=scenario["type"],
                                          epsilon=scenario["epsilon"],
                                          eta=scenario["eta"])
            
            for method_name, method_info in methods.items():
                method_config = GGMConfig(**config.__dict__)
                method_config.gamma = method_info["gamma"]
                method_config.seed = rep + 2000 * scenarios.index(scenario) + \
                                     (0 if method_info["prior"] == "laplace" else 10000) + \
                                     int(method_info["gamma"] * 1000)
                
                if method_info["prior"] == "laplace":
                    fit_result = fit_laplace_with_uncertainty(Y, method_config)
                else:
                    fit_result = fit_map_data_adaptive(Y, method_config)
                
                if fit_result["success"] and fit_result.get("Omega_samples") is not None:
                    rmse, al, cp = compute_uncertainty_metrics(fit_result["Omega_samples"], Omega_true)
                    results[method_name][scenario["name"]].append({
                        "rep": rep, "rmse": rmse, "al": al, "cp": cp
                    })
        
        if (rep + 1) % 10 == 0:
            print(f" ✓")
    
    # Print results summary
    print(f"\n\n📈 Results Summary - {graph_name} (p={p})")
    print(f"{'─'*80}")
    
    for metric, label in [("rmse", "RMSE (↓ better)"), ("cp", "Coverage (target 0.95)"), ("al", "Avg Length (↓ better)")]:
        print(f"\n{label}")
        print(f"{'Scenario':<15}", end="")
        for method in methods.keys():
            print(f" {method:<20}", end="")
        print()
        print(f"{'─'*15}", end="")
        for _ in methods.keys():
            print(f"{'─'*20}", end="")
        print()
        
        for scenario in scenarios:
            print(f"{scenario['name']:<15}", end="")
            for method in methods.keys():
                data = results[method][scenario["name"]]
                if data:
                    values = [d[metric] for d in data]
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    if metric == "cp":
                        dev = abs(mean_val - 0.95)
                        print(f" {mean_val:.3f}±{std_val:.3f} (Δ={dev:.3f})", end="")
                    else:
                        print(f" {mean_val:.4f}±{std_val:.4f}", end="")
                else:
                    print(f" {'N/A':<20}", end="")
            print()
    
    # Compare γ=0.02 vs γ=0.01
    print(f"\n\n🔄 Effect of Increasing Robustness (γ=0.02 vs γ=0.01)")
    print(f"{'─'*80}")
    
    for prior_label, prefix in [("Laplace", "Laplace"), ("Horseshoe", "Horseshoe")]:
        print(f"\n{prior_label} Prior:")
        print(f"{'Scenario':<15} {'RMSE Δ (%)':<15} {'CP Δ':<12} {'AL Δ (%)':<15}")
        print(f"{'─'*15} {'─'*15} {'─'*12} {'─'*15}")
        
        for scenario in scenarios:
            data_001 = results[f"{prefix}_γ001"][scenario["name"]]
            data_002 = results[f"{prefix}_γ002"][scenario["name"]]
            
            if data_001 and data_002:
                rmse_001 = np.mean([d["rmse"] for d in data_001])
                rmse_002 = np.mean([d["rmse"] for d in data_002])
                rmse_change = (rmse_002 - rmse_001) / (rmse_001 + 1e-8) * 100
                
                cp_001 = np.mean([d["cp"] for d in data_001])
                cp_002 = np.mean([d["cp"] for d in data_002])
                cp_change = cp_002 - cp_001
                
                al_001 = np.mean([d["al"] for d in data_001])
                al_002 = np.mean([d["al"] for d in data_002])
                al_change = (al_002 - al_001) / (al_001 + 1e-8) * 100
                
                sig = ""
                if len(data_001) > 1 and len(data_002) > 1:
                    t_stat, p_val = stats.ttest_ind([d["rmse"] for d in data_002],
                                                     [d["rmse"] for d in data_001])
                    if p_val < 0.05:
                        sig = " *" if p_val < 0.05 else "**" if p_val < 0.01 else "***" if p_val < 0.001 else ""
                
                print(f"{scenario['name']:<15} {rmse_change:+6.2f}%{sig:<3}     {cp_change:+.3f}        {al_change:+6.2f}%")
    
    return results


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("🔬 UNCERTAINTY COMPARISON: Laplace vs Horseshoe Priors")
    print("📊 Focus: RMSE, Coverage Probability (CP), Average Length (AL)")
    print("⚙️  Comparing γ = 0.01 vs γ = 0.02")
    print("📐 Structures: AR(1) and AR(2)")
    print("🎯 Dimensions: p = 12 and p = 50")
    print("🔄 Repetitions: 30 per configuration")
    print("="*80)
    
    # Base configuration
    base_config = GGMConfig(
        p=12,
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
    
    # Define all experiments
    experiments = [
        (12, "AR1", generate_ar1_precision(12)),
        (12, "AR2", generate_ar2_precision(12)),
        (50, "AR1", generate_ar1_precision(50)),
        (50, "AR2", generate_ar2_precision(50))
    ]
    
    all_results = {}
    
    for p, graph_name, Omega_true in experiments:
        config = GGMConfig(**base_config.__dict__)
        config.p = p
        
        results = run_uncertainty_comparison(Omega_true, config, n_reps=30, graph_name=graph_name)
        all_results[f"{graph_name}_p{p}"] = results
    
    # Save results
    print("\n\n💾 Saving results...")
    
    with open("uncertainty_comparison_full.pkl", "wb") as f:
        pickle.dump(all_results, f)
    
    # Create summary dataframe
    summary_rows = []
    for exp_name, exp_results in all_results.items():
        for method, scenarios in exp_results.items():
            for scenario, data_list in scenarios.items():
                if data_list:
                    row = {
                        "Experiment": exp_name,
                        "Method": method,
                        "Scenario": scenario,
                        "N_success": len(data_list),
                        "RMSE_mean": np.mean([d["rmse"] for d in data_list]),
                        "RMSE_sd": np.std([d["rmse"] for d in data_list]),
                        "CP_mean": np.mean([d["cp"] for d in data_list]),
                        "CP_sd": np.std([d["cp"] for d in data_list]),
                        "AL_mean": np.mean([d["al"] for d in data_list]),
                        "AL_sd": np.std([d["al"] for d in data_list])
                    }
                    summary_rows.append(row)
    
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("uncertainty_comparison_summary.csv", index=False)
    
    # Create pivot tables for easy reading
    for metric in ["RMSE", "CP", "AL"]:
        pivot = df_summary.pivot_table(
            values=f"{metric}_mean",
            index=["Experiment", "Scenario"],
            columns="Method",
            aggfunc="first"
        ).round(4)
        pivot.to_csv(f"uncertainty_comparison_{metric.lower()}_pivot.csv")
    
    print("\n✅ Analysis complete!")
    print(f"\n📁 Saved files:")
    print(f"   • uncertainty_comparison_full.pkl - Complete results")
    print(f"   • uncertainty_comparison_summary.csv - Summary table")
    print(f"   • uncertainty_comparison_*.csv - Pivot tables per metric")
    
    # Final recommendations
    print("\n" + "="*80)
    print("📋 KEY INSIGHTS:")
    print("="*80)
    print("""
    RMSE (Root Mean Square Error):
        • Measures point estimate accuracy (lower = better)
        • Negative Δ indicates γ=0.02 improves accuracy
    
    CP (Coverage Probability):
        • Measures uncertainty calibration (target = 0.95)
        • CP < 0.95: Under-confident (intervals too narrow)
        • CP > 0.95: Over-confident (intervals too wide)
        • Closer to 0.95 = better calibration
    
    AL (Average Length):
        • Measures precision of uncertainty estimates
        • Lower = better, but only if CP is well-calibrated
        • Trade-off: shorter intervals with good coverage = efficient
    
    Comparison (γ=0.02 vs γ=0.01):
        • Negative RMSE/AL changes = γ=0.02 improves accuracy/efficiency
        • CP changes near zero = similar calibration
        • Significant improvements (*,**,***) indicate statistical significance
    """)
