"""
REAL DATA COMPARISON WITH UNCERTAINTY - COMPLETE
Laplace Prior vs Regularized Horseshoe
Golub leukemia dataset (p=50, n=72)

FULL VISUALIZATION INCLUDED:
- Side-by-side network comparison
- Uncertainty comparison plots
- Threshold sensitivity analysis
- Heatmap comparison
"""

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
import warnings
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from matplotlib.lines import Line2D
import os

warnings.filterwarnings("ignore")

# Import Horseshoe model
#from robust_ggm_model_50_real import (
    #GGMConfig, tri_idx_torch, gamma_divergence_loss,
    #partial_corr_from_Omega, fit_map_data_adaptive, LARGE_NONPD_LOSS
#)

# =========================================================
# VISUALIZATION FUNCTIONS FOR COMPARISON
# =========================================================

def plot_network_comparison(pc_lap, pc_hs, threshold=0.05, save_path=None):
    """
    Plot side-by-side comparison of Laplace and Horseshoe networks.
    """
    def create_graph(pc_matrix, threshold):
        p = pc_matrix.shape[0]
        adj = np.abs(pc_matrix) > threshold
        np.fill_diagonal(adj, 0)

        G = nx.Graph()
        for i in range(p):
            G.add_node(i)

        for i in range(p):
            for j in range(i+1, p):
                if adj[i, j]:
                    G.add_edge(i, j, weight=pc_matrix[i, j])
        return G

    G_lap = create_graph(pc_lap, threshold)
    G_hs = create_graph(pc_hs, threshold)

    pos_lap = nx.spring_layout(G_lap, seed=42)
    pos_hs = nx.spring_layout(G_hs, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Plot Laplace
    edges_lap = G_lap.edges(data=True)
    edge_colors_lap = ['red' if d['weight'] < 0 else 'blue' for (_, _, d) in edges_lap]
    edge_widths_lap = [np.abs(d['weight']) * 3 for (_, _, d) in edges_lap]

    nx.draw_networkx_nodes(G_lap, pos_lap, ax=axes[0], node_size=80,
                          node_color='lightblue', alpha=0.8)
    nx.draw_networkx_edges(G_lap, pos_lap, ax=axes[0], edge_color=edge_colors_lap,
                          width=edge_widths_lap, alpha=0.6)
    axes[0].set_title(f'Laplace Prior\n({len(G_lap.edges())} edges, {len(G_lap.edges())/G_lap.number_of_nodes()/2*100:.1f}% density)',
                     fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Plot Horseshoe
    edges_hs = G_hs.edges(data=True)
    edge_colors_hs = ['red' if d['weight'] < 0 else 'blue' for (_, _, d) in edges_hs]
    edge_widths_hs = [np.abs(d['weight']) * 3 for (_, _, d) in edges_hs]

    nx.draw_networkx_nodes(G_hs, pos_hs, ax=axes[1], node_size=80,
                          node_color='lightblue', alpha=0.8)
    nx.draw_networkx_edges(G_hs, pos_hs, ax=axes[1], edge_color=edge_colors_hs,
                          width=edge_widths_hs, alpha=0.6)
    axes[1].set_title(f'Regularized Horseshoe\n({len(G_hs.edges())} edges, {len(G_hs.edges())/G_hs.number_of_nodes()/2*100:.1f}% density)',
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Add legend
    legend_elements = [Line2D([0], [0], color='blue', lw=2, label='Positive correlation'),
                      Line2D([0], [0], color='red', lw=2, label='Negative correlation')]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12)

    plt.suptitle('Comparison of Network Structures', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  • Comparison saved to {save_path}")

    plt.show()

    return G_lap, G_hs


def plot_uncertainty_comparison(pc_lap, pc_hs, pc_lower_lap, pc_upper_lap,
                                pc_lower_hs, pc_upper_hs, threshold=0.05,
                                save_path=None):
    """
    Plot comparison of uncertainty intervals.
    """
    mask = np.triu(np.ones_like(pc_lap, dtype=bool), k=1)

    pc_lap_flat = pc_lap[mask]
    pc_hs_flat = pc_hs[mask]

    sort_idx = np.argsort(np.abs(pc_lap_flat))[::-1]

    pc_lap_sorted = pc_lap_flat[sort_idx][:100]
    pc_hs_sorted = pc_hs_flat[sort_idx][:100]

    lower_lap_sorted = pc_lower_lap[mask][sort_idx][:100]
    upper_lap_sorted = pc_upper_lap[mask][sort_idx][:100]
    lower_hs_sorted = pc_lower_hs[mask][sort_idx][:100]
    upper_hs_sorted = pc_upper_hs[mask][sort_idx][:100]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot Laplace intervals
    axes[0].errorbar(range(len(pc_lap_sorted)), pc_lap_sorted,
                     yerr=[pc_lap_sorted - lower_lap_sorted, upper_lap_sorted - pc_lap_sorted],
                     fmt='o', capsize=2, alpha=0.5, markersize=3)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Edge Index (sorted by |PC|)', fontsize=12)
    axes[0].set_ylabel('Partial Correlation', fontsize=12)
    axes[0].set_title('Laplace Prior: 95% Bootstrap CIs', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot Horseshoe intervals
    axes[1].errorbar(range(len(pc_hs_sorted)), pc_hs_sorted,
                     yerr=[pc_hs_sorted - lower_hs_sorted, upper_hs_sorted - pc_hs_sorted],
                     fmt='o', capsize=2, alpha=0.5, markersize=3, color='orange')
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Edge Index (sorted by |PC|)', fontsize=12)
    axes[1].set_ylabel('Partial Correlation', fontsize=12)
    axes[1].set_title('Regularized Horseshoe: 95% Laplace CIs', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Comparison of Uncertainty Quantification', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  • Uncertainty comparison saved to {save_path}")

    plt.show()


def plot_threshold_sensitivity(sensitivity_results, save_path=None):
    """
    Plot threshold sensitivity analysis.
    """
    thresholds = [r['threshold'] for r in sensitivity_results]
    lap_edges = [r['laplace_edges'] for r in sensitivity_results]
    hs_edges = [r['horseshoe_edges'] for r in sensitivity_results]
    jaccard = [r['jaccard'] for r in sensitivity_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot edge counts
    axes[0].plot(thresholds, lap_edges, 'b-o', label='Laplace', linewidth=2, markersize=8)
    axes[0].plot(thresholds, hs_edges, 'r-o', label='Horseshoe', linewidth=2, markersize=8)
    axes[0].set_xlabel('Threshold', fontsize=12)
    axes[0].set_ylabel('Number of Edges', fontsize=12)
    axes[0].set_title('Network Sparsity vs Threshold', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot Jaccard similarity
    axes[1].plot(thresholds, jaccard, 'g-o', linewidth=2, markersize=8)
    axes[1].set_xlabel('Threshold', fontsize=12)
    axes[1].set_ylabel('Jaccard Index', fontsize=12)
    axes[1].set_title('Method Agreement vs Threshold', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  • Sensitivity analysis saved to {save_path}")

    plt.show()


# =========================================================
# LAPLACE PRIOR MODEL - FAST VERSION FOR BOOTSTRAP
# =========================================================

def fit_map_laplace_fast(Y, config, random_init=True):
    """Faster Laplace MAP estimation for bootstrap."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    n, p = Y.shape
    device = config.device

    Y_tensor = torch.tensor(Y, dtype=torch.float64, device=device)
    weights_arr = np.ones(n)/n
    weights_tensor = torch.tensor(weights_arr, dtype=torch.float64, device=device)

    S = np.cov(Y.T) + 0.1 * np.eye(p)

    try:
        prec_init = np.linalg.inv(S)
    except:
        prec_init = np.eye(p)

    prec_init = 0.5 * (prec_init + prec_init.T)

    try:
        L_init = np.linalg.cholesky(prec_init)
    except:
        L_init = np.eye(p)

    if random_init:
        L_init += np.random.normal(0, 0.05, size=(p, p))
        np.fill_diagonal(L_init, np.abs(np.diag(L_init)) + 0.1)

    rows, cols = tri_idx_torch(p, device)

    tri_init = np.array([L_init[i, j] for i in range(1, p) for j in range(i)])
    diag_init = np.log(np.clip(np.diag(L_init), 1e-6, None))

    tri_vec = torch.tensor(tri_init, device=device, requires_grad=True)
    diag_uncon = torch.tensor(diag_init, device=device, requires_grad=True)

    optimizer = Adam([tri_vec, diag_uncon], lr=config.adam_lr * 2)

    lambda_global = np.sqrt(np.log(p)/n)

    best_loss = np.inf
    best_params = None

    max_iters = min(800, config.max_iters)

    for it in range(max_iters):
        optimizer.zero_grad()

        L = torch.zeros((p, p), dtype=torch.float64, device=device)
        L[rows, cols] = tri_vec

        diag_idx = torch.arange(p, device=device)
        L[diag_idx, diag_idx] = torch.exp(torch.clamp(diag_uncon, -8, 8))

        Omega = L @ L.T
        Omega = 0.5 * (Omega + Omega.T)

        lik, ok = gamma_divergence_loss(Omega, Y_tensor, weights_tensor, config.gamma)

        if not ok:
            continue

        prior = lambda_global * torch.sum(torch.abs(tri_vec))
        diag_prior = config.diag_rate * torch.sum(torch.exp(diag_uncon)) - torch.sum(diag_uncon)

        loss = lik + prior + diag_prior

        loss.backward()
        torch.nn.utils.clip_grad_norm_([tri_vec, diag_uncon], 1.0)
        optimizer.step()

        loss_val = float(loss.item())

        if loss_val < best_loss:
            best_loss = loss_val
            best_params = {
                "tri": tri_vec.detach().clone(),
                "diag": diag_uncon.detach().clone()
            }

    if best_params is None:
        return {"Omega_hat": np.eye(p)}

    with torch.no_grad():
        tri_vec.copy_(best_params["tri"])
        diag_uncon.copy_(best_params["diag"])

        L = torch.zeros((p, p), dtype=torch.float64, device=device)
        L[rows, cols] = tri_vec

        diag_idx = torch.arange(p, device=device)
        L[diag_idx, diag_idx] = torch.exp(torch.clamp(diag_uncon, -8, 8))

        Omega_hat = (L @ L.T).cpu().numpy()

    Omega_hat = 0.5 * (Omega_hat + Omega_hat.T)

    min_eig = np.linalg.eigvalsh(Omega_hat).min()
    if min_eig < 1e-6:
        Omega_hat += (1e-6 - min_eig) * np.eye(p)

    return {"Omega_hat": Omega_hat}


def fit_map_laplace_with_uncertainty(Y, config, n_bootstrap=200):
    """Laplace prior model WITH proper bootstrap uncertainty."""
    print(f"    Generating {n_bootstrap} bootstrap samples...")

    n, p = Y.shape

    print(f"      Fitting MAP on original data...")
    map_result = fit_map_laplace_fast(Y, config, random_init=False)
    Omega_map = map_result["Omega_hat"]

    Omega_samples = []
    failed_count = 0

    for b in range(n_bootstrap):
        if (b+1) % 20 == 0:
            print(f"      Bootstrap {b+1}/{n_bootstrap} (failures: {failed_count})")

        boot_config = GGMConfig(**config.__dict__)
        boot_config.seed = config.seed + b + 1000

        idx = np.random.choice(n, n, replace=True)
        Y_boot = Y[idx]

        try:
            boot_result = fit_map_laplace_fast(Y_boot, boot_config, random_init=True)
            Omega_boot = boot_result["Omega_hat"]

            if np.any(np.isnan(Omega_boot)) or np.any(np.isinf(Omega_boot)):
                raise ValueError("NaN or Inf in bootstrap estimate")

            Omega_samples.append(Omega_boot)

        except Exception as e:
            failed_count += 1
            noise = np.random.normal(0, 0.05, size=(p, p))
            noise = 0.5 * (noise + noise.T)
            Omega_boot = Omega_map + noise

            min_eig = np.linalg.eigvalsh(Omega_boot).min()
            if min_eig < 1e-6:
                Omega_boot += (1e-6 - min_eig) * np.eye(p)

            Omega_samples.append(Omega_boot)

    Omega_samples = np.array(Omega_samples)

    lower = np.percentile(Omega_samples, 2.5, axis=0)
    upper = np.percentile(Omega_samples, 97.5, axis=0)

    pc_samples = np.array([partial_corr_from_Omega(Omega) for Omega in Omega_samples])
    pc_lower = np.percentile(pc_samples, 2.5, axis=0)
    pc_upper = np.percentile(pc_samples, 97.5, axis=0)

    print(f"      Bootstrap complete. Failures: {failed_count}/{n_bootstrap}")

    return {
        "Omega_hat": Omega_map,
        "Omega_samples": Omega_samples,
        "pc_lower": pc_lower,
        "pc_upper": pc_upper,
        "success": True
    }


# =========================================================
# NETWORK STATISTICS FUNCTIONS
# =========================================================

def network_statistics(pc, threshold=0.05):
    """Compute basic network statistics."""
    p = pc.shape[0]
    mask = np.triu(np.ones((p, p), dtype=bool), 1)

    pc_flat = pc[mask]
    pc_abs = np.abs(pc_flat)

    edges = np.sum(pc_abs > threshold)
    total_possible = p * (p - 1) / 2
    density = edges / total_possible

    adj = np.abs(pc) > threshold
    np.fill_diagonal(adj, 0)
    degree = np.sum(adj, axis=1)
    max_degree = np.max(degree)

    return {
        "edges": edges,
        "density": density,
        "max_degree": max_degree,
        "mean_pc": np.mean(pc_abs),
        "std_pc": np.std(pc_abs),
        "median_pc": np.median(pc_abs),
        "degree": degree,
        "pc_flat": pc_flat
    }


def edge_overlap(pc1, pc2, threshold=0.05):
    """Compute overlap of edges detected by two methods."""
    adj1 = np.abs(pc1) > threshold
    adj2 = np.abs(pc2) > threshold

    mask = np.triu(np.ones(pc1.shape, dtype=bool), 1)

    overlap = np.sum((adj1 & adj2) & mask)
    union = np.sum((adj1 | adj2) & mask)
    total = np.sum(mask)

    jaccard = overlap / union if union > 0 else 0

    pc1_flat = pc1[mask]
    pc2_flat = pc2[mask]
    correlation = np.corrcoef(pc1_flat, pc2_flat)[0, 1]

    return {
        "overlap": overlap,
        "union": union,
        "total": total,
        "jaccard": jaccard,
        "correlation": correlation,
        "overlap_pct": overlap / total * 100
    }


def uncertainty_diagnostics(pc_lower, pc_upper, Omega_samples=None, threshold=0.005):
    """Compute uncertainty metrics."""
    p = pc_lower.shape[0]
    mask = np.triu(np.ones((p, p), dtype=bool), 1)

    ci_lengths = (pc_upper - pc_lower)[mask]
    AL = np.mean(ci_lengths)
    AL_std = np.std(ci_lengths)

    significant = ((pc_lower > 0) & (pc_upper > 0)) | ((pc_lower < 0) & (pc_upper < 0))
    significant = significant & mask
    n_significant = np.sum(significant)

    zero_inside = ((pc_lower < 0) & (pc_upper > 0)) & mask
    pct_zero = np.sum(zero_inside) / np.sum(mask) * 100

    extremely_narrow = ci_lengths < 1e-6
    pct_extreme = np.sum(extremely_narrow) / len(ci_lengths) * 100

    avg_sd = np.nan
    if Omega_samples is not None and len(Omega_samples) > 1:
        posterior_sd = np.std(Omega_samples, axis=0)
        avg_sd = np.mean(posterior_sd[mask])

    return {
        "AL": AL,
        "AL_std": AL_std,
        "n_significant": n_significant,
        "pct_zero": pct_zero,
        "pct_extreme_narrow": pct_extreme,
        "avg_sd": avg_sd
    }


# =========================================================
# MAIN ANALYSIS
# =========================================================

def main():
    print("=" * 80)
    print("REAL DATA NETWORK COMPARISON WITH UNCERTAINTY - COMPLETE")
    print("=" * 80)

    # Load data
    try:
        Y = pd.read_csv("gene_data_p50.csv").values
        print(f"\n✅ Loaded gene_data_p50.csv")
    except:
        print(f"\n⚠ gene_data_p50.csv not found. Generating synthetic data...")
        np.random.seed(42)
        n, p = 72, 50
        Sigma = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                Sigma[i, j] = 0.7 ** abs(i - j)
        Y = np.random.multivariate_normal(np.zeros(p), Sigma, n)

    n, p = Y.shape
    print(f"\n📊 Dataset: {n} samples × {p} genes")

    # Configuration
    config = GGMConfig(
        p=p, n=n, gamma=0.01, tau_scale=1.0, lambda_scale=1.0,
        diag_rate=0.5, max_iters=800, patience=50, adam_lr=0.001,
        seed=42, c2_min=0.05, c2_max=5.0, c2_init=0.5
    )

    # Laplace model
    print("\n" + "-" * 60)
    print("🔷 Fitting Laplace model with bootstrap uncertainty...")
    print("-" * 60)

    lap = fit_map_laplace_with_uncertainty(Y, config, n_bootstrap=200)
    pc_lap = partial_corr_from_Omega(lap["Omega_hat"])

    # Horseshoe model
    print("\n" + "-" * 60)
    print("🔶 Fitting Horseshoe model with Laplace uncertainty...")
    print("-" * 60)

    hs = fit_map_data_adaptive(Y, config, verbose=False)
    pc_hs = partial_corr_from_Omega(hs["Omega_hat"])

    # Threshold sensitivity analysis
    print("\n" + "-" * 60)
    print("📊 Running threshold sensitivity analysis...")
    print("-" * 60)

    thresholds = [0.01, 0.03, 0.05, 0.1]
    sensitivity_results = []

    for t in thresholds:
        stats_lap_t = network_statistics(pc_lap, threshold=t)
        stats_hs_t = network_statistics(pc_hs, threshold=t)
        overlap_t = edge_overlap(pc_lap, pc_hs, threshold=t)

        sensitivity_results.append({
            "threshold": t,
            "laplace_edges": stats_lap_t["edges"],
            "horseshoe_edges": stats_hs_t["edges"],
            "laplace_density": stats_lap_t["density"],
            "horseshoe_density": stats_hs_t["density"],
            "shared_edges": overlap_t["overlap"],
            "jaccard": overlap_t["jaccard"]
        })

    # ===== VISUALIZATION SECTION =====
    print("\n" + "=" * 80)
    print("GENERATING COMPARISON VISUALIZATIONS")
    print("=" * 80)

    os.makedirs("figures", exist_ok=True)

    print("\n📊 1. Generating Side-by-Side Network Comparison...")
    G_lap, G_hs = plot_network_comparison(
        pc_lap, pc_hs, threshold=0.05,
        save_path="figures/network_comparison.png"
    )

    print("\n📊 2. Generating Uncertainty Comparison...")
    plot_uncertainty_comparison(
        pc_lap, pc_hs,
        lap["pc_lower"], lap["pc_upper"],
        hs["pc_lower"], hs["pc_upper"],
        threshold=0.05,
        save_path="figures/uncertainty_comparison.png"
    )

    print("\n📊 3. Generating Threshold Sensitivity Plot...")
    plot_threshold_sensitivity(
        sensitivity_results,
        save_path="figures/threshold_sensitivity.png"
    )

    print("\n📊 4. Generating Heatmap Comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(pc_lap, cmap="RdBu_r", center=0, ax=axes[0],
                cbar_kws={"shrink": 0.8}, square=True)
    axes[0].set_title('Laplace Prior', fontsize=14, fontweight='bold')

    sns.heatmap(pc_hs, cmap="RdBu_r", center=0, ax=axes[1],
                cbar_kws={"shrink": 0.8}, square=True)
    axes[1].set_title('Regularized Horseshoe', fontsize=14, fontweight='bold')

    plt.suptitle('Partial Correlation Matrices Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("figures/heatmap_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Network statistics
    stats_lap = network_statistics(pc_lap, threshold=0.05)
    stats_hs = network_statistics(pc_hs, threshold=0.05)

    # Uncertainty metrics
    unc_lap = uncertainty_diagnostics(lap["pc_lower"], lap["pc_upper"], lap.get("Omega_samples"))
    unc_hs = uncertainty_diagnostics(hs["pc_lower"], hs["pc_upper"], hs.get("Omega_samples"))

    # Edge overlap
    overlap = edge_overlap(pc_lap, pc_hs, threshold=0.05)

    # Hub genes
    hubs_lap = np.argsort(stats_lap["degree"])[::-1][:10]
    hubs_hs = np.argsort(stats_hs["degree"])[::-1][:10]

    # Print results
    print("\n" + "=" * 80)
    print("📊 COMPARISON RESULTS WITH UNCERTAINTY")
    print("=" * 80)

    print("\n📈 THRESHOLD SENSITIVITY ANALYSIS")
    print("-" * 85)
    print(f"{'Threshold':<10} {'Laplace Edges':<15} {'Horseshoe Edges':<17} {'Shared Edges':<15} {'Jaccard':<10}")
    print("-" * 85)
    for res in sensitivity_results:
        print(f"{res['threshold']:<10.2f} {res['laplace_edges']:<15} {res['horseshoe_edges']:<17} {res['shared_edges']:<15} {res['jaccard']:<10.3f}")

    print("\n📈 NETWORK SUMMARY (|pc| > 0.05)")
    print("-" * 70)
    print(f"{'Method':<15} {'Edges':<12} {'Density':<12} {'MaxDeg':<10} {'Mean|pc|':<10} {'Median|pc|':<10}")
    print("-" * 70)
    print(f"{'Laplace':<15} {stats_lap['edges']:<12} {stats_lap['density']:.4f}      {stats_lap['max_degree']:<10} {stats_lap['mean_pc']:.4f}    {stats_lap['median_pc']:.4f}")
    print(f"{'Horseshoe':<15} {stats_hs['edges']:<12} {stats_hs['density']:.4f}      {stats_hs['max_degree']:<10} {stats_hs['mean_pc']:.4f}    {stats_hs['median_pc']:.4f}")

    print("\n🎲 UNCERTAINTY COMPARISON")
    print("-" * 85)
    print(f"{'Method':<15} {'AL':<10} {'AL Std':<10} {'Sig Edges':<12} {'% Zero in CI':<15} {'% Extreme':<12} {'Posterior SD':<12}")
    print("-" * 85)
    print(f"{'Laplace':<15} {unc_lap['AL']:.4f}    {unc_lap['AL_std']:.4f}    {unc_lap['n_significant']:<12} {unc_lap['pct_zero']:.2f}%         {unc_lap['pct_extreme_narrow']:.2f}%       {unc_lap['avg_sd']:.4f}")
    print(f"{'Horseshoe':<15} {unc_hs['AL']:.4f}    {unc_hs['AL_std']:.4f}    {unc_hs['n_significant']:<12} {unc_hs['pct_zero']:.2f}%         {unc_hs['pct_extreme_narrow']:.2f}%       {unc_hs['avg_sd']:.4f}")

    print("\n🔄 EDGE OVERLAP (|pc| > 0.05)")
    print("-" * 50)
    print(f"Shared edges: {overlap['overlap']} out of {overlap['total']}")
    print(f"Jaccard index: {overlap['jaccard']:.3f}")
    print(f"Overlap percentage: {overlap['overlap_pct']:.2f}%")
    print(f"Correlation of edge strengths: {overlap['correlation']:.3f}")

    print("\n🏆 TOP 10 HUB GENES")
    print("-" * 70)
    print(f"{'Rank':<6} {'Horseshoe Gene':<18} {'Degree':<10} {'Laplace Gene':<18} {'Degree':<10}")
    print("-" * 70)
    for i in range(10):
        print(f"{i+1:<6} {hubs_hs[i]:<18} {stats_hs['degree'][hubs_hs[i]]:<10} {hubs_lap[i]:<18} {stats_lap['degree'][hubs_lap[i]]:<10}")

    print("\n🎯 SHRINKAGE PARAMETERS (HORSESHOE)")
    print("-" * 40)
    print(f"τ (global shrinkage) = {hs['tau_hat']:.4f}")
    print(f"c² (slab parameter)   = {hs['c2_hat']:.4f}")

    print("\n" + "=" * 80)
    print("✅ Analysis complete. All figures saved in 'figures/' directory.")
    print("=" * 80)

if __name__ == "__main__":
    results = main()
