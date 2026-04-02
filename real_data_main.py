"""
COMPLETE IMPLEMENTATION: Robust Gaussian Graphical Model
with γ-divergence, Data-Adaptive Regularized Horseshoe Prior
with hyperprior on c² (slab parameter)

Works with real gene expression data (n=72, p=50)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import math
import warnings
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from matplotlib.lines import Line2D
import os

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==================== CORE CONFIGURATION ====================
@dataclass
class GGMConfig:
    """Configuration for Robust GGM with data-adaptive c²."""
    p: int = 50
    n: int = 72
    gamma: float = 0.01
    tau_scale: float = 1.0
    lambda_scale: float = 1.0
    diag_rate: float = 0.5
    max_iters: int = 800
    patience: int = 50
    adam_lr: float = 0.001
    B_laplace_samples: int = 500
    seed: int = 42
    c2_min: float = 0.05
    c2_max: float = 5.0
    c2_init: float = 0.5
    device: Optional[torch.device] = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.set_default_dtype(torch.float64)
LARGE_NONPD_LOSS = 1e8

# ==================== VISUALIZATION FUNCTIONS ====================
def plot_network(pc_matrix, threshold=0.05, title="Network", figsize=(12, 12),
                 save_path=None, show_labels=False):
    """
    Plot network graph from partial correlation matrix.
    """
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

    pos = nx.spring_layout(G, seed=42)
    edges = G.edges(data=True)

    edge_colors = ['red' if d['weight'] < 0 else 'blue' for (_, _, d) in edges]
    edge_widths = [np.abs(d['weight']) * 3 for (_, _, d) in edges]

    plt.figure(figsize=figsize)

    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue',
                           alpha=0.8, edgecolors='black', linewidths=0.5)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths,
                           alpha=0.6, edge_cmap=plt.cm.RdBu)

    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')

    legend_elements = [Line2D([0], [0], color='blue', lw=2, label='Positive correlation'),
                      Line2D([0], [0], color='red', lw=2, label='Negative correlation')]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  • Network saved to {save_path}")

    plt.show()

    n_edges = len(G.edges())
    n_possible = p * (p - 1) // 2
    density = n_edges / n_possible * 100
    n_positive = sum(1 for c in edge_colors if c == 'blue')
    n_negative = sum(1 for c in edge_colors if c == 'red')

    print(f"\n  Network Statistics (threshold={threshold}):")
    print(f"  • Number of edges: {n_edges}")
    print(f"  • Network density: {density:.2f}%")
    print(f"  • Positive edges: {n_positive}")
    print(f"  • Negative edges: {n_negative}")

    return G, pos


def plot_heatmap(pc_matrix, title="Heatmap", figsize=(12, 10),
                 vmin=None, vmax=None, cmap="RdBu_r", save_path=None):
    """
    Plot heatmap of partial correlation matrix.
    """
    plt.figure(figsize=figsize)

    max_abs = np.max(np.abs(pc_matrix))
    if vmin is None:
        vmin = -max_abs
    if vmax is None:
        vmax = max_abs

    sns.heatmap(pc_matrix, cmap=cmap, center=0,
                vmin=vmin, vmax=vmax,
                square=True, cbar_kws={"shrink": 0.8, "label": "Partial Correlation"})

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Gene Index", fontsize=12)
    plt.ylabel("Gene Index", fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  • Heatmap saved to {save_path}")

    plt.show()

    mask = np.triu(np.ones_like(pc_matrix, dtype=bool), k=1)
    pc_flat = pc_matrix[mask]
    print(f"\n  Partial Correlation Statistics:")
    print(f"  • Mean: {np.mean(pc_flat):.4f}")
    print(f"  • Std: {np.std(pc_flat):.4f}")
    print(f"  • Min: {np.min(pc_flat):.4f}")
    print(f"  • Max: {np.max(pc_flat):.4f}")


def plot_degree_distribution(pc_matrix, threshold=0.05, title="Degree Distribution",
                            figsize=(10, 6), save_path=None):
    """
    Plot degree distribution of the network.
    """
    adj = np.abs(pc_matrix) > threshold
    np.fill_diagonal(adj, 0)
    degree = np.sum(adj, axis=1)

    plt.figure(figsize=figsize)

    counts, bins, patches = plt.hist(degree, bins='auto', edgecolor='black',
                                      alpha=0.7, density=True)

    from scipy import stats
    try:
        degree_pos = degree[degree > 0]
        if len(degree_pos) > 10:
            params = stats.expon.fit(degree_pos)
            x = np.linspace(0, max(degree), 100)
            pdf = stats.expon.pdf(x, *params)
            plt.plot(x, pdf, 'r-', label=f'Exponential fit (λ={1/params[1]:.2f})', linewidth=2)
    except:
        pass

    plt.xlabel("Degree (number of connections)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  • Degree distribution saved to {save_path}")

    plt.show()

    print(f"\n  Degree Statistics (threshold={threshold}):")
    print(f"  • Mean degree: {np.mean(degree):.2f}")
    print(f"  • Median degree: {np.median(degree):.2f}")
    print(f"  • Max degree: {np.max(degree)}")
    print(f"  • Min degree: {np.min(degree)}")

    return degree


def plot_correlation_distribution(pc_matrix, title="Partial Correlation Distribution",
                                  figsize=(10, 6), save_path=None):
    """
    Plot distribution of partial correlation coefficients.
    """
    mask = np.triu(np.ones_like(pc_matrix, dtype=bool), k=1)
    pc_flat = pc_matrix[mask]

    plt.figure(figsize=figsize)

    plt.hist(pc_flat, bins=50, edgecolor='black', alpha=0.7, density=True,
             label='Observed')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Zero')

    from scipy import stats
    mu, std = stats.norm.fit(pc_flat)
    x = np.linspace(min(pc_flat), max(pc_flat), 100)
    plt.plot(x, stats.norm.pdf(x, mu, std), 'g-',
             label=f'Normal fit (μ={mu:.3f}, σ={std:.3f})', linewidth=2)

    plt.xlabel("Partial Correlation", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  • Distribution saved to {save_path}")

    plt.show()

    print(f"\n  Correlation Statistics:")
    print(f"  • Mean: {np.mean(pc_flat):.4f}")
    print(f"  • Std: {np.std(pc_flat):.4f}")
    print(f"  • Skewness: {stats.skew(pc_flat):.4f}")
    print(f"  • Kurtosis: {stats.kurtosis(pc_flat):.4f}")

    return pc_flat


def plot_hub_subnetwork(pc_matrix, hub_indices, threshold=0.05, figsize=(10, 8), save_path=None):
    """
    Plot subnetwork focusing on hub genes.
    """
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

    pos = nx.spring_layout(G, seed=42)

    node_colors = ['red' if node in hub_indices else 'lightblue' for node in G.nodes()]
    node_sizes = [300 if node in hub_indices else 80 for node in G.nodes()]

    plt.figure(figsize=figsize)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)

    edges = G.edges(data=True)
    edge_colors = ['red' if d['weight'] < 0 else 'blue' for (_, _, d) in edges]
    edge_widths = [np.abs(d['weight']) * 3 for (_, _, d) in edges]
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, alpha=0.6)

    hub_labels = {node: f"Gene {node}" for node in hub_indices}
    nx.draw_networkx_labels(G, pos, labels=hub_labels, font_size=10, font_weight='bold')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markersize=10, label='Hub Genes'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
               markersize=10, label='Other Genes'),
        Line2D([0], [0], color='blue', lw=2, label='Positive correlation'),
        Line2D([0], [0], color='red', lw=2, label='Negative correlation')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.title(f'Hub Gene Subnetwork (Top {len(hub_indices)} Hubs)', fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  • Hub subnetwork saved to {save_path}")

    plt.show()

    return G


# ==================== CORE UTILITIES ====================
_TRI_CACHE = {}
def tri_idx_torch(p: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cached indices for efficiency."""
    key = (p, str(device))
    if key in _TRI_CACHE:
        return _TRI_CACHE[key]

    rows, cols = [], []
    for i in range(1, p):
        for j in range(i):
            rows.append(i)
            cols.append(j)

    result = (
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(cols, dtype=torch.long, device=device)
    )
    _TRI_CACHE[key] = result
    return result

def partial_corr_from_Omega(Omega: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Compute partial correlations from precision matrix."""
    Omega = 0.5 * (Omega + Omega.T)
    diag = np.diag(Omega).copy()
    diag[diag <= 0] = eps
    D = np.sqrt(diag)

    with np.errstate(divide='ignore', invalid='ignore'):
        pc = -Omega / (np.outer(D, D) + eps)

    np.fill_diagonal(pc, 0.0)
    return 0.5 * (pc + pc.T)

def build_precision_matrix(tri_vec: torch.Tensor, diag_uncon: torch.Tensor,
                          rows: torch.Tensor, cols: torch.Tensor, p: int) -> torch.Tensor:
    """Build precision matrix from triangular and diagonal parameters."""
    L = torch.zeros((p, p), dtype=torch.float64, device=tri_vec.device)
    L[rows, cols] = tri_vec
    diag_idx = torch.arange(p, device=tri_vec.device)
    L[diag_idx, diag_idx] = torch.exp(torch.clamp(diag_uncon, -8.0, 8.0))
    Omega = L @ L.T
    return 0.5 * (Omega + Omega.T)

# ==================== DATA-ADAPTIVE REGULARIZED HORSESHOE ====================
def regularized_horseshoe_data_adaptive(
    beta: torch.Tensor,
    lambda_uncon: torch.Tensor,
    tau_uncon: torch.Tensor,
    c2_uncon: torch.Tensor,
    tau_scale: float = 1.0,
    lambda_scale: float = 1.0,
    eps: float = 1e-12,
    c2_min: float = 0.05,
    c2_max: float = 5.0
) -> Tuple[torch.Tensor, float, float]:
    """
    Data-adaptive regularized horseshoe with hyperprior on c².
    """
    lambda_pos = F.softplus(lambda_uncon) + eps
    tau_pos = F.softplus(tau_uncon) + eps
    c2_raw = F.softplus(c2_uncon) + eps
    c2 = torch.clamp(c2_raw, c2_min, c2_max)

    lambda2 = lambda_pos ** 2
    tau2 = tau_pos ** 2

    denom = c2 + tau2 * lambda2 + eps
    tilde_lambda2 = c2 * lambda2 / denom
    var_beta = tau2 * tilde_lambda2 + eps

    beta_prior = torch.sum(0.5 * torch.log(var_beta) + 0.5 * (beta ** 2) / var_beta)

    half_cauchy_const = torch.log(beta.new_tensor(np.pi / 2.0))

    lambda_prior = torch.sum(
        half_cauchy_const
        + torch.log(beta.new_tensor(lambda_scale))
        + torch.log1p((lambda_pos / lambda_scale) ** 2)
    )

    tau_prior = (
        half_cauchy_const
        + torch.log(beta.new_tensor(tau_scale))
        + torch.log1p((tau_pos / tau_scale) ** 2)
    )

    alpha, beta_param = 2.0, 1.0
    c2_prior = (alpha + 1) * torch.log(c2) + beta_param / c2

    jac_lambda = -torch.sum(torch.log(torch.clamp(torch.sigmoid(lambda_uncon), min=eps)))
    jac_tau = -torch.sum(torch.log(torch.clamp(torch.sigmoid(tau_uncon), min=eps)))
    jac_c2 = -torch.sum(torch.log(torch.clamp(torch.sigmoid(c2_uncon), min=eps)))

    total_prior = beta_prior + lambda_prior + tau_prior + c2_prior + jac_lambda + jac_tau + jac_c2

    return total_prior, float(tau_pos.item()), float(c2.item())

# ==================== γ-DIVERGENCE LIKELIHOOD ====================
def gamma_divergence_loss(Omega: torch.Tensor,
                         Y: torch.Tensor,
                         weights: torch.Tensor,
                         gamma: float,
                         jitter: float = 1e-5) -> Tuple[torch.Tensor, bool]:
    """
    γ-divergence negative log-likelihood with weighted centering.
    """
    if abs(gamma) < 1e-12:
        return gaussian_likelihood_loss(Omega, Y, weights, jitter)

    n, p = Y.shape
    device = Omega.device

    w = weights / (weights.sum() + 1e-30)
    mu = torch.sum(w.unsqueeze(1) * Y, dim=0)
    Yc = Y - mu

    Omega_eps = Omega + jitter * torch.eye(p, device=device, dtype=Omega.dtype)
    Omega_eps = 0.5 * (Omega_eps + Omega_eps.T)

    try:
        L = torch.linalg.cholesky(Omega_eps)
        logdet = 2 * torch.sum(torch.log(torch.diag(L)))
    except:
        return torch.tensor(LARGE_NONPD_LOSS, device=device, dtype=Omega.dtype), False

    quad = torch.einsum("ni,ij,nj->n", Yc, Omega_eps, Yc)
    a = -0.5 * gamma * quad

    m = torch.max(a)
    log_sum = m + torch.log(torch.sum(w * torch.exp(a - m)) + 1e-30)

    negloglik = -(1.0 / gamma) * log_sum - (1.0 / (2.0 * (1.0 + gamma))) * logdet
    return negloglik, True

def gaussian_likelihood_loss(Omega: torch.Tensor,
                            Y: torch.Tensor,
                            weights: torch.Tensor,
                            jitter: float = 1e-5) -> Tuple[torch.Tensor, bool]:
    """
    Standard Gaussian negative log-likelihood with weighted centering.
    """
    n, p = Y.shape
    device = Omega.device

    w = weights / (weights.sum() + 1e-30)
    mu = torch.sum(w.unsqueeze(1) * Y, dim=0)
    Yc = Y - mu
    Yw = Yc * torch.sqrt(w).unsqueeze(1)

    S_w = (Yw.T @ Yw)
    S_w = 0.5 * (S_w + S_w.T)

    Omega_eps = Omega + jitter * torch.eye(p, device=device, dtype=Omega.dtype)
    Omega_eps = 0.5 * (Omega_eps + Omega_eps.T)

    try:
        L = torch.linalg.cholesky(Omega_eps)
        logdet = 2 * torch.sum(torch.log(torch.diag(L)))
    except:
        return torch.tensor(LARGE_NONPD_LOSS, device=device, dtype=Omega.dtype), False

    negloglik = -logdet + torch.trace(S_w @ Omega_eps)
    return negloglik, True

# ==================== TOTAL LOSS WITH DATA-ADAPTIVE C² ====================
def total_loss_data_adaptive(
    Y: torch.Tensor,
    tri_vec: torch.Tensor,
    diag_uncon: torch.Tensor,
    lambda_uncon: torch.Tensor,
    tau_uncon: torch.Tensor,
    c2_uncon: torch.Tensor,
    config: GGMConfig,
    weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, bool, Dict]:
    """
    Total loss with data-adaptive c² and stability safeguards.
    """
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

    rhs_prior, tau_val, c2_val = regularized_horseshoe_data_adaptive(
        tri_vec, lambda_uncon, tau_uncon, c2_uncon,
        tau_scale=config.tau_scale,
        lambda_scale=config.lambda_scale,
        c2_min=config.c2_min,
        c2_max=config.c2_max
    )

    diag_prior = config.diag_rate * torch.sum(L_diag) - torch.sum(diag_uncon)

    total_loss = lik + rhs_prior + diag_prior

    details = {
        "Omega": Omega.detach(),
        "likelihood": float(lik.item()),
        "rhs_prior": float(rhs_prior.item()),
        "diag_prior": float(diag_prior.item()),
        "tau": tau_val,
        "c2": c2_val,
        "L_diag": L_diag.detach()
    }

    return total_loss, True, details

# ==================== LAPLACE APPROXIMATION UTILITIES ====================
def compute_laplace_covariance_fast(
    Y_tensor: torch.Tensor,
    tri_vec: torch.Tensor,
    diag_uncon: torch.Tensor,
    lambda_uncon: torch.Tensor,
    tau_uncon: torch.Tensor,
    c2_uncon: torch.Tensor,
    config: GGMConfig,
    weights_tensor: torch.Tensor
) -> np.ndarray:
    """Compute Laplace covariance using efficient Hessian computation."""
    p = config.p
    M = p * (p - 1) // 2

    theta = torch.cat([tri_vec.detach(), diag_uncon.detach()])
    theta = theta.clone().requires_grad_(True)

    def objective(theta_vec):
        tri = theta_vec[:M]
        diag = theta_vec[M:]

        loss, ok, _ = total_loss_data_adaptive(
            Y_tensor,
            tri,
            diag,
            lambda_uncon.detach(),
            tau_uncon.detach(),
            c2_uncon.detach(),
            config,
            weights_tensor
        )
        return loss

    loss = objective(theta)
    grad = torch.autograd.grad(loss, theta, create_graph=True)[0]

    dim = theta.shape[0]
    H = torch.zeros(dim, dim, dtype=torch.float64, device=theta.device)

    for i in range(dim):
        grad2 = torch.autograd.grad(grad[i], theta, retain_graph=True, allow_unused=True)[0]
        if grad2 is not None:
            H[i] = grad2
        else:
            H[i, i] = 1.0

    H_np = H.detach().cpu().numpy()
    H_np = 0.5 * (H_np + H_np.T)

    ridge = 1e-6 * np.eye(dim)
    H_stab = H_np + ridge

    try:
        cov = np.linalg.inv(H_stab)
    except np.linalg.LinAlgError:
        ridge = 1e-4 * np.eye(dim)
        cov = np.linalg.inv(H_stab + ridge)

    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals[eigvals < 1e-10] = 1e-10
    cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return cov

def sample_laplace_posterior(
    theta_map: np.ndarray,
    cov_laplace: np.ndarray,
    p: int,
    n_samples: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample from Laplace posterior approximation."""
    dim = theta_map.shape[0]
    M = p * (p - 1) // 2

    try:
        L = np.linalg.cholesky(cov_laplace)
    except np.linalg.LinAlgError:
        ridge = 1e-8 * np.eye(dim)
        cov_reg = cov_laplace + ridge
        L = np.linalg.cholesky(cov_reg)

    z = np.random.randn(n_samples, dim)
    theta_samples = theta_map + z @ L.T

    Omega_samples = []
    for s in theta_samples:
        tri = s[:M]
        diag = s[M:]

        L_mat = np.zeros((p, p))
        k = 0
        for i in range(1, p):
            for j in range(i):
                L_mat[i, j] = tri[k]
                k += 1

        L_mat[np.diag_indices(p)] = np.exp(np.clip(diag, -8.0, 8.0))
        Omega = L_mat @ L_mat.T
        Omega = 0.5 * (Omega + Omega.T)
        Omega_samples.append(Omega)

    return np.array(Omega_samples), theta_samples

def compute_pc_uncertainty(
    Omega_samples: np.ndarray,
    ci_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute partial correlation uncertainty from posterior samples."""
    pc_samples = np.array([
        partial_corr_from_Omega(Omega)
        for Omega in Omega_samples
    ])

    pc_mean = np.mean(pc_samples, axis=0)
    alpha = (1 - ci_level) / 2
    pc_lower = np.percentile(pc_samples, alpha * 100, axis=0)
    pc_upper = np.percentile(pc_samples, (1 - alpha) * 100, axis=0)

    return pc_mean, pc_lower, pc_upper

# ==================== MAP ESTIMATION WITH DATA-ADAPTIVE C² ====================
def fit_map_data_adaptive(
    Y: np.ndarray,
    config: GGMConfig,
    weights: Optional[np.ndarray] = None,
    verbose: bool = False
) -> Dict:
    """MAP estimation with data-adaptive c² and Laplace uncertainty."""
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
    M = p * (p - 1) // 2

    tri_init = np.array([L_init[i, j] for i in range(1, p) for j in range(i)])
    diag_init = np.log(np.clip(np.diag(L_init), 1e-6, None))

    tri_vec = torch.tensor(tri_init, device=device, requires_grad=True)
    diag_uncon = torch.tensor(diag_init, device=device, requires_grad=True)
    lambda_uncon = torch.zeros(M, device=device, requires_grad=True)

    tau_init = 0.2 * config.tau_scale
    tau_uncon_init = np.log(np.exp(tau_init) - 1.0)
    tau_uncon = torch.tensor(tau_uncon_init, dtype=torch.float64, device=device, requires_grad=True)

    c2_init = config.c2_init
    c2_uncon_init = np.log(np.exp(c2_init) - 1.0)
    c2_uncon = torch.tensor(c2_uncon_init, dtype=torch.float64, device=device, requires_grad=True)

    params = [tri_vec, diag_uncon, lambda_uncon, tau_uncon, c2_uncon]

    def compute_loss():
        return total_loss_data_adaptive(
            Y_tensor, tri_vec, diag_uncon, lambda_uncon, tau_uncon, c2_uncon,
            config, weights_tensor
        )

    optimizer = Adam([
        {"params": [tri_vec, diag_uncon], "lr": config.adam_lr},
        {"params": [lambda_uncon, tau_uncon, c2_uncon], "lr": 2.0 * config.adam_lr},
    ], eps=1e-8)

    loss_trace, tau_trace, c2_trace = [], [], []
    best_loss = float("inf")
    best_params = None
    patience_counter = 0

    for it in range(config.max_iters):
        optimizer.zero_grad()
        loss, ok, details = compute_loss()
        if not ok:
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        loss_val = float(loss.item())
        loss_trace.append(loss_val)
        tau_trace.append(details["tau"])
        c2_trace.append(details["c2"])

        if loss_val < best_loss - 1e-6:
            best_loss = loss_val
            best_params = {
                "tri_vec": tri_vec.detach().clone(),
                "diag_uncon": diag_uncon.detach().clone(),
                "lambda_uncon": lambda_uncon.detach().clone(),
                "tau_uncon": tau_uncon.detach().clone(),
                "c2_uncon": c2_uncon.detach().clone(),
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > config.patience:
            if verbose:
                print(f"Early stopping at iteration {it}")
            break

    if best_params is not None:
        with torch.no_grad():
            tri_vec.copy_(best_params["tri_vec"])
            diag_uncon.copy_(best_params["diag_uncon"])
            lambda_uncon.copy_(best_params["lambda_uncon"])
            tau_uncon.copy_(best_params["tau_uncon"])
            c2_uncon.copy_(best_params["c2_uncon"])

    with torch.no_grad():
        L = torch.zeros((p, p), dtype=torch.float64, device=device)
        L[rows, cols] = tri_vec
        diag_idx = torch.arange(p, device=device)
        L[diag_idx, diag_idx] = torch.exp(torch.clamp(diag_uncon, -8.0, 8.0))
        Omega_hat = (L @ L.T).cpu().numpy()
        Omega_hat = 0.5 * (Omega_hat + Omega_hat.T)

        _, _, final_details = compute_loss()

    if verbose:
        print("\nComputing Laplace approximation for uncertainty...")
        print(f"  • Parameter dimension: {M + p}")

    cov_laplace = compute_laplace_covariance_fast(
        Y_tensor, tri_vec, diag_uncon, lambda_uncon, tau_uncon, c2_uncon,
        config, weights_tensor
    )

    eigvals = np.linalg.eigvalsh(cov_laplace)
    smallest_eig = np.min(eigvals)
    largest_eig = np.max(eigvals)
    condition_number = largest_eig / max(smallest_eig, 1e-10)

    if verbose:
        print(f"  • Smallest eigenvalue: {smallest_eig:.2e}")
        print(f"  • Largest eigenvalue: {largest_eig:.2e}")
        print(f"  • Condition number: {condition_number:.2e}")

    theta_map = torch.cat([tri_vec.detach(), diag_uncon.detach()]).cpu().numpy()
    laplace_se = np.sqrt(np.diag(cov_laplace))

    Omega_samples, theta_samples = sample_laplace_posterior(
        theta_map, cov_laplace, p, n_samples=config.B_laplace_samples
    )

    pc_mean, pc_lower, pc_upper = compute_pc_uncertainty(Omega_samples)

    if verbose:
        print(f"\nEstimated Parameters:")
        print(f"  • τ (global shrinkage) = {final_details['tau']:.4f}")
        print(f"  • c² (slab parameter)   = {final_details['c2']:.4f}")
        print(f"\nLaplace Approximation:")
        print(f"  • Parameter dimension: {len(theta_map)}")
        print(f"  • Posterior samples: {config.B_laplace_samples}")
        print(f"  • Avg SE: {np.mean(laplace_se):.6f}")

    return {
        "Omega_hat": Omega_hat,
        "tau_hat": final_details["tau"],
        "c2_hat": final_details["c2"],
        "c2_trace": c2_trace,
        "loss_trace": loss_trace,
        "tau_trace": tau_trace,
        "final_likelihood": final_details["likelihood"],
        "final_prior": final_details["rhs_prior"],
        "final_diag_prior": final_details["diag_prior"],
        "final_total_loss": best_loss,
        "iterations": len(loss_trace),
        "success": best_params is not None,
        "laplace_cov": cov_laplace,
        "laplace_se": laplace_se,
        "theta_map": theta_map,
        "Omega_samples": Omega_samples,
        "pc_mean": pc_mean,
        "pc_lower": pc_lower,
        "pc_upper": pc_upper,
        "laplace_diagnostics": {
            "smallest_eig": smallest_eig,
            "largest_eig": largest_eig,
            "condition_number": condition_number
        }
    }

# ==================== EVALUATION METRICS ====================
def compute_edge_uncertainty(pc_lower, pc_upper, threshold=0.005):
    """Compute uncertainty-aware edge detection metrics."""
    p = pc_lower.shape[0]
    mask = np.triu(np.ones((p, p), dtype=bool), k=1)

    significant = ((pc_lower > threshold) & (pc_upper > threshold)) | \
                  ((pc_lower < -threshold) & (pc_upper < -threshold))
    significant = significant & mask

    ci_width = pc_upper - pc_lower
    avg_ci_width = np.mean(ci_width[mask])

    zero_in_ci = ((pc_lower < 0) & (pc_upper > 0)) & mask
    pct_zero_in_ci = np.sum(zero_in_ci) / np.sum(mask) * 100

    return {
        "n_significant_edges": np.sum(significant),
        "avg_ci_width": avg_ci_width,
        "significant_mask": significant,
        "pct_zero_in_ci": pct_zero_in_ci
    }

def identify_hub_genes(pc_matrix, threshold=0.05, top_k=10):
    """Identify hub genes based on degree."""
    adj = np.abs(pc_matrix) > threshold
    np.fill_diagonal(adj, 0)
    degree = np.sum(adj, axis=1)
    top_indices = np.argsort(degree)[::-1][:top_k]

    return {
        "degree": degree,
        "top_hubs": top_indices,
        "top_degrees": degree[top_indices]
    }

# ==================== MAIN DEMONSTRATION ====================
def main():
    """Demonstrate data-adaptive c² estimation with Laplace uncertainty."""
    print("=" * 70)
    print("ROBUST GAUSSIAN GRAPHICAL MODEL")
    print("WITH DATA-ADAPTIVE C² (SLAB PARAMETER)")
    print("AND LAPLACE UNCERTAINTY QUANTIFICATION")
    print("=" * 70)
    print("\nRUNNING ON REAL GENE EXPRESSION DATA")
    print("=" * 70)

    print("\nLoading gene expression data...")
    try:
        Y_real = pd.read_csv("gene_data_p50.csv").values
        n_real, p_real = Y_real.shape
        print(f"  • Dataset shape: {n_real} samples × {p_real} genes")
        print(f"  • Sample size: n = {n_real}")
        print(f"  • Number of genes: p = {p_real}")
    except FileNotFoundError:
        print("  ⚠ Warning: gene_data_p50.csv not found. Using simulated data.")
        p_real = 50
        n_real = 72
        np.random.seed(42)
        # Create AR-2 structure for simulation
        Omega_true = np.eye(p_real)
        for i in range(1, p_real):
            Omega_true[i, i-1] = Omega_true[i-1, i] = -0.5
        for i in range(2, p_real):
            Omega_true[i, i-2] = Omega_true[i-2, i] = -0.25
        Omega_true += 0.1 * np.eye(p_real)
        Sigma_true = np.linalg.inv(Omega_true)
        Y_real = np.random.multivariate_normal(np.zeros(p_real), Sigma_true, n_real)
        print(f"  • Using simulated AR-2 data: {n_real} samples × {p_real} genes")

    config = GGMConfig(
        p=p_real,
        n=n_real,
        gamma=0.01,
        tau_scale=1.0,
        lambda_scale=1.0,
        diag_rate=0.5,
        max_iters=800,
        patience=50,
        adam_lr=0.001,
        B_laplace_samples=100,
        seed=42,
        c2_min=0.05,
        c2_max=5.0,
        c2_init=0.5
    )

    print(f"\nConfiguration:")
    print(f"  • p = {config.p}, n = {config.n}")
    print(f"  • γ = {config.gamma}")
    print(f"  • c² range = [{config.c2_min}, {config.c2_max}]")
    print(f"  • Laplace samples = {config.B_laplace_samples}")

    print(f"\n" + "=" * 70)
    print("FITTING MODEL ON REAL GENE EXPRESSION DATA")
    print("=" * 70)

    res_real = fit_map_data_adaptive(Y_real, config, verbose=True)

    print(f"\n" + "=" * 70)
    print("GENE REGULATORY NETWORK ANALYSIS")
    print("=" * 70)

    pc_real = partial_corr_from_Omega(res_real['Omega_hat'])

    # ===== VISUALIZATION SECTION =====
    print(f"\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    os.makedirs("figures", exist_ok=True)

    print("\n📊 1. Generating Network Graph...")
    G, pos = plot_network(
        pc_real, threshold=0.05,
        title="Gene Regulatory Network - Regularized Horseshoe Prior",
        figsize=(12, 12), save_path="figures/horseshoe_network.png"
    )

    print("\n📊 2. Generating Partial Correlation Heatmap...")
    plot_heatmap(
        pc_real,
        title="Partial Correlation Matrix - Regularized Horseshoe Prior",
        figsize=(12, 10), save_path="figures/horseshoe_heatmap.png"
    )

    print("\n📊 3. Generating Degree Distribution...")
    degree = plot_degree_distribution(
        pc_real, threshold=0.05,
        title="Network Degree Distribution",
        save_path="figures/horseshoe_degree_distribution.png"
    )

    print("\n📊 4. Generating Partial Correlation Distribution...")
    pc_flat = plot_correlation_distribution(
        pc_real,
        title="Distribution of Partial Correlations",
        save_path="figures/horseshoe_correlation_distribution.png"
    )

    print("\n📊 5. Generating Hub Gene Subnetwork...")
    hub_analysis = identify_hub_genes(pc_real, threshold=0.05, top_k=10)
    plot_hub_subnetwork(
        pc_real, hub_analysis['top_hubs'][:5], threshold=0.05,
        save_path="figures/hub_subnetwork.png"
    )

    # Continue with analysis
    pc_flat_full = pc_real[np.triu_indices_from(pc_real, k=1)]
    n_edges_possible = len(pc_flat_full)

    print(f"\nNetwork Statistics:")
    print(f"  • Total possible edges: {n_edges_possible}")
    print(f"  • Mean |partial correlation|: {np.mean(np.abs(pc_flat_full)):.4f}")
    print(f"  • Std |partial correlation|: {np.std(np.abs(pc_flat_full)):.4f}")
    print(f"  • Max |partial correlation|: {np.max(np.abs(pc_flat_full)):.4f}")

    print(f"\nThreshold Analysis:")
    for threshold in [0.05, 0.1, 0.15, 0.2]:
        n_edges = np.sum(np.abs(pc_flat_full) > threshold)
        density = n_edges / n_edges_possible * 100
        print(f"  • Edges (|pc| > {threshold:.2f}): {n_edges} ({density:.1f}% density)")

    print(f"\nTop 10 Hub Genes (most connected):")
    print(f"{'Rank':<6} {'Gene Index':<12} {'Degree':<10}")
    print("-" * 30)
    for i, (gene_idx, degree_val) in enumerate(zip(
        hub_analysis['top_hubs'],
        hub_analysis['top_degrees']
    )):
        print(f"{i+1:<6} {gene_idx:<12} {degree_val:<10}")

    print(f"\n" + "=" * 70)
    print("UNCERTAINTY QUANTIFICATION (LAPLACE APPROXIMATION)")
    print("=" * 70)

    if res_real['success']:
        print(f"\nLaplace Results:")
        print(f"  • Parameter dimension: {len(res_real['theta_map'])}")
        print(f"  • Avg standard error: {np.mean(res_real['laplace_se']):.6f}")

        edge_uncertainty = compute_edge_uncertainty(
            res_real['pc_lower'], res_real['pc_upper']
        )
        print(f"\n  Edge Uncertainty (95% Credible Intervals):")
        print(f"    • Significant edges: {edge_uncertainty['n_significant_edges']}")
        print(f"    • Avg CI width: {edge_uncertainty['avg_ci_width']:.4f}")

    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
    Key findings from gene expression data (n={n_real}, p={p_real}):

    • Global shrinkage (τ): {res_real['tau_hat']:.3f}
    • Slab parameter (c²): {res_real['c2_hat']:.3f}
    • Network density (|pc| > 0.05): {np.sum(np.abs(pc_flat_full) > 0.05)} edges
    • Significant edges (95% CI): {edge_uncertainty['n_significant_edges'] if res_real['success'] else 'N/A'}
    """)

    return {
        "real_data": res_real,
        "config": config,
        "Y_real": Y_real,
        "pc_real": pc_real,
        "hub_analysis": hub_analysis,
        "visualizations": {"G": G, "degree": degree, "pc_flat": pc_flat}
    }

if __name__ == "__main__":
    results = main()
