"""
Hydro-code EoS timing benchmark — with autograd graph (derivatives).

Measures the realistic pipeline cost for a hydro code that needs EoS
derivatives (∂Pe/∂ρ, ∂Pe/∂T, ∂Se/∂T) at every cell per timestep.

Pipeline per batch
------------------
  1. Z-scale all conditions
  2. Forward pass — phi(x), graph retained (use_graph=True)
  3. Post-processing — FD integrals, trapz → Pe, Ee, Se, μe, Qe
  4. Backward pass — scalar reduce then .backward() giving ∂Pe/∂ρ,
     ∂Pe/∂T, ∂Se/∂T simultaneously from the same graph
  5. Graph discarded

Staged timing: forward, post-processing, backward, total.

Batch sizes
-----------
Matches the no-graph benchmark (helmholtz_scaling_benchmark.py) up to the
OOM ceiling:  100, 400, 1024, 4096, then doubling until OOM.

Outputs
-------
  plots/helmholtz_graph_benchmark/
    scaling_total_time.{pdf,png}     — log-log: total/fwd/post/bwd vs N
    scaling_per_condition.{pdf,png}  — µs/cond vs batch size
    scaling_breakdown.{pdf,png}      — stacked bar fwd/post/bwd
    comparison.{pdf,png}             — with-graph vs no-graph per-cond cost
    helmholtz_graph_results.npz      — raw timing arrays

Usage
-----
    python -m scripts.helmholtz_graph_benchmark \\
        --config configs/helmholtz_pipeline.yaml --device cuda
"""

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_loader import load_eos_config, load_pinn, build_x_grid
from src.inputs import (
    z_scale_inputs,
    compute_gamma,
    compute_lambda,
    r0_from_density,
    r0_from_density_torch,       # new
    z_scale_inputs_torch,        # new
    compute_gamma_torch,         # new
    compute_lambda_torch,        # new
    B_M,
    KEV_TO_J,
    M_PROTON,
    C1,
)
from src.fd_integrals import fermi_dirac_half, fermi_dirac_three_half
from src.quantities.internal_energy import C_K, C_UEN, C_PRESSURE

# ---------------------------------------------------------------------------
# Fixed element
# ---------------------------------------------------------------------------

Z = 26.0   # Iron
A = 56.0

# ---------------------------------------------------------------------------
# Reference ranges — same as no-graph benchmark, no extrapolation
# ---------------------------------------------------------------------------

RHO_MIN, RHO_MAX = 0.2996, 54.62    # g/cm³
T_MIN,   T_MAX   = 0.1476, 14.660   # keV

# Batch sizes — start at 100, double until OOM.
# Mirror the no-graph sizes at the low end for direct comparison.
BATCH_SIZES = [
    (10,  10),    #   100
    (20,  20),    #   400
    (32,  32),    #  1024
    (64,  64),    #  4096
    (100, 100),   # 10000
    (128, 128),   # 16384
    (160, 160),   # 25600
    (200, 200),   # 40000  — may OOM; script skips gracefully
]

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------

N_WARMUP = 5
N_TIMED  = 20

# No-graph results path for comparison plot
NOGRAPH_NPZ = (
    Path(__file__).resolve().parents[1]
    / "plots" / "helmholtz_scaling_extended" / "helmholtz_scaling_results.npz"
)

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _sync(device: str):
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()

def _tick(device: str) -> float:
    _sync(device)
    return time.perf_counter()

def _tock(t0: float, device: str) -> float:
    _sync(device)
    return (time.perf_counter() - t0) * 1e3   # ms

# ---------------------------------------------------------------------------
# Condition parameter preparation
# ---------------------------------------------------------------------------

def _make_grids(n_rho: int, n_T: int):
    rho_grid = np.logspace(np.log10(RHO_MIN), np.log10(RHO_MAX), n_rho)
    T_grid   = np.logspace(np.log10(T_MIN),   np.log10(T_MAX),   n_T)
    return rho_grid, T_grid


def _build_condition_params(rho_list, T_list):
    conditions = []
    for rho in rho_list:
        for T_keV in T_list:
            r0           = r0_from_density(rho, A)
            alpha_1, T_1 = z_scale_inputs(Z, r0, T_keV)
            gamma        = compute_gamma(T_1)
            lam          = compute_lambda(alpha_1, T_1)
            r_0_1        = alpha_1 * B_M
            V_1          = (4.0 / 3.0) * math.pi * r_0_1**3
            conditions.append((rho, T_keV, alpha_1, T_1, gamma, lam, r_0_1, V_1))  # added rho, T_keV

    def _col(i, dt=np.float32):
        return np.array([c[i] for c in conditions], dtype=dt)

    return (
        _col(0),               # rho_arr      — NEW
        _col(1),               # T_keV_arr    — NEW
        _col(2),               # alpha_1_arr
        _col(3),               # T_1_arr
        _col(4),               # gamma_arr
        _col(5),               # lam_arr
        _col(6, np.float64),   # r_0_1_arr
        _col(7, np.float64),   # V_1_arr
    )

# ---------------------------------------------------------------------------
# Single timed pass — graph retained through backward
# ---------------------------------------------------------------------------

def _run_once(model, inputs_base_t, x_t,
              rho_t, T_keV_t,
              gamma_t, lam_t, r_0_1_t, V_1_t, T_1_t,
              device, n_cond, n_x):
    """
    One full EoS pass with autograd: forward → post-processing → backward.

    rho_t and T_t_phys are leaf tensors that require_grad so we can
    differentiate Pe, Se w.r.t. them.  The whole chain from these inputs
    through alpha_1, T_1, gamma, lam, xi, phi, FD integrals to Pe/Se is
    kept in the graph; a single .backward() then gives all derivatives.

    Returns (t_forward_ms, t_post_ms, t_backward_ms).
    """
    # Leaf tensors — autograd graph starts from physical (rho, T)
    rho_leaf   = rho_t.detach().requires_grad_(True)     # [N_COND]
    T_keV_leaf = T_keV_t.detach().requires_grad_(True)   # [N_COND]

    # Z-scale in torch — keeps graph connected to rho_leaf, T_keV_leaf
    r0_leaf          = r0_from_density_torch(rho_leaf, A)
    alpha_1_l, T_1_l = z_scale_inputs_torch(Z, r0_leaf, T_keV_leaf)
    gamma_l          = compute_gamma_torch(T_1_l)
    lam_l            = compute_lambda_torch(alpha_1_l, T_1_l)
    r_0_1_l          = alpha_1_l * B_M
    V_1_l            = (4.0 / 3.0) * math.pi * r_0_1_l ** 3

    # REPLACE the inputs_t construction block with:
    x_col = inputs_base_t[:, 0:1].detach()   # [N_COND*N_X, 1] — x, no grad needed

    alpha_1_rep = alpha_1_l.repeat_interleave(n_x).unsqueeze(1)  # [N_COND*N_X, 1]
    T_1_rep     = T_1_l.repeat_interleave(n_x).unsqueeze(1)      # [N_COND*N_X, 1]

    # inputs_t is NOT a leaf — it's derived from rho_leaf/T_keV_leaf via alpha_1_l, T_1_l
    # but it still has requires_grad=True because alpha_1_l and T_1_l do
    inputs_t = torch.cat([x_col, alpha_1_rep, T_1_rep], dim=1)   # [N_COND*N_X, 3]


    # ------------------------------------------------------------------
    # Stage 2: Forward pass — graph retained
    # ------------------------------------------------------------------
    t0_fwd = _tick(device)
    phi_flat = model(inputs_t)           # [N_COND*N_X, 1] — graph intact
    phi_t    = phi_flat.reshape(n_cond, n_x)   # [N_COND, N_X] — do NOT detach
    t_forward = _tock(t0_fwd, device)

    # ------------------------------------------------------------------
    # Stage 3: Post-processing — all in-graph
    # ------------------------------------------------------------------
    t0_post = _tick(device)

    # xi: [N_COND, N_X]
    xi_t   = gamma_l.unsqueeze(1) * phi_t / (lam_l.unsqueeze(1) * x_t.unsqueeze(0))
    xi_1_t = xi_t[:, -1]                        # [N_COND]

    # FD integrals on full grid
    xi_col_t   = xi_t.unsqueeze(2)              # [N_COND, N_X, 1]
    F32_t      = fermi_dirac_three_half(xi_col_t).squeeze(2)   # [N_COND, N_X]
    F12_t      = fermi_dirac_half(xi_col_t).squeeze(2)         # [N_COND, N_X]

    # Boundary FD integrals: [N_COND]
    xi_1_col_t = xi_1_t.unsqueeze(1).unsqueeze(2)
    F32_b_t    = fermi_dirac_three_half(xi_1_col_t).squeeze(2).squeeze(1)
    F12_b_t    = fermi_dirac_half(xi_1_col_t).squeeze(2).squeeze(1)

    # Volume integrals
    trap_K   = torch.trapezoid(F32_t * x_t.unsqueeze(0)**2, x_t, dim=1)
    trap_Uen = torch.trapezoid(F12_t * x_t.unsqueeze(0),    x_t, dim=1)

    # K_1, U_en_1, U_ee_1  [N_COND]
    K_1_t    =  C_K   * r_0_1_l**3 * T_1_l**2.5 * trap_K
    U_en_1_t = -C_UEN * r_0_1_l**2 * T_1_l**1.5 * trap_Uen
    P_e1_t   =  C_PRESSURE * T_1_l**2.5 * F32_b_t
    U_ee_1_t =  3.0 * P_e1_t * V_1_l - 2.0 * K_1_t - U_en_1_t

    # Chemical potential μe  [N_COND]  [J]
    mu_e1_t  = xi_1_t * T_1_l * KEV_TO_J

    # Pressure Pe [N_COND] [Pa]
    Pe_t = Z**(10.0 / 3.0) * P_e1_t

    # Internal energy Ee [N_COND] [J/atom]
    E_Z1_t = K_1_t + U_en_1_t + U_ee_1_t
    Ee_t   = Z**(7.0 / 3.0) * E_Z1_t

    # Entropy Se [N_COND] [dimensionless, S/k_B per atom in Z=1 then Z-scaled]
    kT_1_t  = T_1_l * KEV_TO_J
    S_Z1_t  = ((5.0 / 3.0) * K_1_t - mu_e1_t + U_en_1_t + 2.0 * U_ee_1_t) / kT_1_t
    Se_t    = Z * S_Z1_t

    # Charge state Qe [N_COND] [dimensionless]
    # Q_1 = C_Q * r_0_1^3 * T_1^(3/2) * F_{1/2}(xi_1)
    C_Q_const = C1 * (4.0 / 3.0) * math.pi * KEV_TO_J**1.5
    Q_1_t  = C_Q_const * r_0_1_l**3 * T_1_l**1.5 * F12_b_t
    Qe_t   = Z * Q_1_t

    t_postproc = _tock(t0_post, device)

    # ------------------------------------------------------------------
    # Stage 4: Backward pass — derive ∂Pe/∂inputs, ∂Pe/∂T, ∂Se/∂T
    # A single scalar reduction + .backward() populates .grad on inputs_t,
    # giving all partial derivatives simultaneously at every condition.
    # ------------------------------------------------------------------
    t0_bwd = _tick(device)

    # Scalar reduce: sum over conditions (representative of a hydro RHS
    # that accumulates residuals over all cells before the backward sweep)
    dPe_drho, dPe_dT = torch.autograd.grad(
        Pe_t.sum(), [rho_leaf, T_keV_leaf],
        create_graph=False, retain_graph=True,
    )
    dSe_dT, = torch.autograd.grad(
        Se_t.sum(), [T_keV_leaf],
        create_graph=False, retain_graph=False,
    )

    t_backward = _tock(t0_bwd, device)

    # Release graph
    del inputs_t, phi_t, xi_t, xi_col_t, F32_t, F12_t

    return t_forward, t_postproc, t_backward

# ---------------------------------------------------------------------------
# Benchmark one batch size — returns None on OOM
# ---------------------------------------------------------------------------

def benchmark_size(model, n_rho, n_T, cfg, device):
    n_x   = cfg.get("n_x",   256)
    x_min = cfg.get("x_min", 1e-6)
    n_cond = n_rho * n_T

    rho_grid, T_grid = _make_grids(n_rho, n_T)
    rho_arr, T_keV_arr, alpha_1_arr, T_1_arr, gamma_arr, lam_arr, r_0_1_arr, V_1_arr = (
        _build_condition_params(rho_grid, T_grid)
    )

    # Build input tensor — uploaded once, zero-copy leaf each run
    x_np      = build_x_grid(n_x, x_min)
    x_rep     = np.tile(x_np.astype(np.float32), n_cond)
    alpha_rep = np.repeat(alpha_1_arr, n_x)
    T_rep     = np.repeat(T_1_arr,     n_x)
    inputs_np = np.stack([x_rep, alpha_rep, T_rep], axis=1)

    try:
        inputs_base_t = torch.from_numpy(inputs_np).to(device)
    except torch.OutOfMemoryError:
        return None

    def _t(arr):
        return torch.from_numpy(arr.astype(np.float32)).to(device)

    x_t      = torch.from_numpy(x_np.astype(np.float32)).to(device)
    gamma_t  = _t(gamma_arr)
    lam_t    = _t(lam_arr)
    r_0_1_t  = _t(r_0_1_arr)
    V_1_t    = _t(V_1_arr)
    T_1_t    = _t(T_1_arr)
    alpha_1_t = _t(alpha_1_arr)
    # ADD after the existing _t(...) calls:
    rho_t    = torch.from_numpy(rho_arr.astype(np.float32)).to(device)
    T_keV_t  = torch.from_numpy(T_keV_arr.astype(np.float32)).to(device)

    args = (model, inputs_base_t, x_t,
            rho_t, T_keV_t,
            gamma_t, lam_t, r_0_1_t, V_1_t, T_1_t,
            device, n_cond, n_x)

    # Warmup
    try:
        for _ in range(N_WARMUP):
            _run_once(*args)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

    # Timed runs
    fwd_times  = np.empty(N_TIMED)
    post_times = np.empty(N_TIMED)
    bwd_times  = np.empty(N_TIMED)
    try:
        for i in range(N_TIMED):
            tf, tp, tb = _run_once(*args)
            fwd_times[i]  = tf
            post_times[i] = tp
            bwd_times[i]  = tb
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

    total_times = fwd_times + post_times + bwd_times
    return {
        "n_cond":        n_cond,
        "fwd_mean":      fwd_times.mean(),    "fwd_std":      fwd_times.std(),
        "post_mean":     post_times.mean(),   "post_std":     post_times.std(),
        "bwd_mean":      bwd_times.mean(),    "bwd_std":      bwd_times.std(),
        "total_mean":    total_times.mean(),  "total_std":    total_times.std(),
        "fwd_pc_mean":   fwd_times.mean()    / n_cond,
        "fwd_pc_std":    fwd_times.std()     / n_cond,
        "post_pc_mean":  post_times.mean()   / n_cond,
        "post_pc_std":   post_times.std()    / n_cond,
        "bwd_pc_mean":   bwd_times.mean()    / n_cond,
        "bwd_pc_std":    bwd_times.std()     / n_cond,
        "total_pc_mean": total_times.mean()  / n_cond,
        "total_pc_std":  total_times.std()   / n_cond,
    }

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot(arrays, plot_dir: Path, device: str):
    n   = arrays["n_cond"]
    col = {"total": "#2176ae", "fwd": "#e07b39", "post": "#57a773", "bwd": "#9b4dca"}

    # ---- Figure 1: log-log total/fwd/post/bwd vs N -------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(n, arrays["total_mean"], yerr=arrays["total_std"],
                fmt="o-",  color=col["total"], label="Total",          capsize=4, lw=2)
    ax.errorbar(n, arrays["fwd_mean"],   yerr=arrays["fwd_std"],
                fmt="s--", color=col["fwd"],   label="Forward pass",   capsize=4, lw=1.5)
    ax.errorbar(n, arrays["post_mean"],  yerr=arrays["post_std"],
                fmt="^--", color=col["post"],  label="Post-processing",capsize=4, lw=1.5)
    ax.errorbar(n, arrays["bwd_mean"],   yerr=arrays["bwd_std"],
                fmt="D--", color=col["bwd"],   label="Backward pass",  capsize=4, lw=1.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of conditions", fontsize=12)
    ax.set_ylabel("Wall time (ms)", fontsize=12)
    ax.set_title(f"Batched Helmholtz + derivatives — scaling  [{device}]", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(plot_dir / f"scaling_total_time.{ext}", dpi=150)
    plt.close(fig)

    # ---- Figure 2: per-condition cost --------------------------------------
    pc_mean = arrays["total_pc_mean"] * 1e3
    pc_std  = arrays["total_pc_std"]  * 1e3
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(n, pc_mean, yerr=pc_std, fmt="o-", color=col["total"], capsize=4, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("Number of conditions", fontsize=12)
    ax.set_ylabel("Time per condition (µs)", fontsize=12)
    ax.set_title(f"Per-condition cost vs batch size  [{device}]", fontsize=12)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(plot_dir / f"scaling_per_condition.{ext}", dpi=150)
    plt.close(fig)

    # ---- Figure 3: stacked bar breakdown -----------------------------------
    x_pos = np.arange(len(n))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x_pos, arrays["fwd_mean"],  0.5, label="Forward",        color=col["fwd"])
    ax.bar(x_pos, arrays["post_mean"], 0.5, bottom=arrays["fwd_mean"],
           label="Post-processing", color=col["post"])
    ax.bar(x_pos, arrays["bwd_mean"],  0.5,
           bottom=arrays["fwd_mean"] + arrays["post_mean"],
           label="Backward", color=col["bwd"])
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(int(v)) for v in n], rotation=30, ha="right")
    ax.set_xlabel("Number of conditions", fontsize=12)
    ax.set_ylabel("Wall time (ms)", fontsize=12)
    ax.set_title(f"Stage breakdown (with graph)  [{device}]", fontsize=12)
    ax.legend(fontsize=10); ax.grid(True, axis="y", ls=":", alpha=0.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(plot_dir / f"scaling_breakdown.{ext}", dpi=150)
    plt.close(fig)

    # ---- Figure 4: comparison with no-graph --------------------------------
    if NOGRAPH_NPZ.exists():
        ng = np.load(NOGRAPH_NPZ)
        ng_n    = ng["n_cond"]
        ng_pc   = ng["total_pc_mean"] * 1e3
        ng_pc_s = ng["total_pc_std"]  * 1e3

        wg_pc   = arrays["total_pc_mean"] * 1e3
        wg_pc_s = arrays["total_pc_std"]  * 1e3

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.errorbar(ng_n, ng_pc, yerr=ng_pc_s, fmt="o-",
                    color="#2176ae", label="No graph (no derivatives)", capsize=4, lw=2)
        ax.errorbar(n,    wg_pc, yerr=wg_pc_s, fmt="s-",
                    color="#9b4dca", label="With graph (∂Pe/∂ρ, ∂Pe/∂T, ∂Se/∂T)", capsize=4, lw=2)
        ax.set_xscale("log")
        ax.set_xlabel("Number of conditions", fontsize=12)
        ax.set_ylabel("Time per condition (µs)", fontsize=12)
        ax.set_title(f"Graph overhead: with vs without derivatives  [{device}]", fontsize=12)
        ax.legend(fontsize=10); ax.grid(True, which="both", ls=":", alpha=0.5)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(plot_dir / f"comparison.{ext}", dpi=150)
        plt.close(fig)
        print(f"  Comparison plot saved (no-graph data: {ng_n.tolist()} conds)")
    else:
        print(f"  [skip] No-graph npz not found at {NOGRAPH_NPZ} — comparison plot skipped")

    print(f"Plots saved to {plot_dir}/")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(cfg, device):
    print(f"\n{'='*70}")
    print(f"Hydro EoS benchmark — with graph + backward  (Iron Z=26, A=56)")
    print(f"  ρ range : [{RHO_MIN}, {RHO_MAX}] g/cm³  (log-spaced)")
    print(f"  T range : [{T_MIN}, {T_MAX}] keV  (log-spaced)")
    print(f"  Device  : {device}")
    print(f"  Warmup  : {N_WARMUP}  |  Timed : {N_TIMED}")
    print(f"  Derivatives: ∂Pe/∂ρ, ∂Pe/∂T, ∂Se/∂T")
    print(f"{'='*70}\n")

    model, _ = load_pinn(cfg, device=device, verbose=False)
    model.eval()
    torch.set_grad_enabled(True)

    results = []
    for n_rho, n_T in BATCH_SIZES:
        n_cond = n_rho * n_T
        print(f"--- {n_rho}×{n_T} = {n_cond:>6} conditions  "
              f"(input tensor [{n_cond * cfg.get('n_x', 256)}, 3]) ---")
        r = benchmark_size(model, n_rho, n_T, cfg, device)
        torch.cuda.empty_cache()

        if r is None:
            print(f"    [OOM] skipped — GPU out of memory at {n_cond} conditions\n")
            break

        results.append(r)
        print(f"    total  : {r['total_mean']:8.2f} ± {r['total_std']:.2f} ms  "
              f"| per-cond: {r['total_pc_mean']*1e3:.4f} ± {r['total_pc_std']*1e3:.4f} µs")
        print(f"    fwd    : {r['fwd_mean']:8.2f} ± {r['fwd_std']:.2f} ms  "
              f"| per-cond: {r['fwd_pc_mean']*1e3:.4f} µs")
        print(f"    post   : {r['post_mean']:8.2f} ± {r['post_std']:.2f} ms  "
              f"| per-cond: {r['post_pc_mean']*1e3:.4f} µs")
        print(f"    bwd    : {r['bwd_mean']:8.2f} ± {r['bwd_std']:.2f} ms  "
              f"| per-cond: {r['bwd_pc_mean']*1e3:.4f} µs")
        print()

    if not results:
        print("No results — all sizes OOM'd.")
        return

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    cw = 8
    print(f"\n{'='*80}")
    print(f"Summary — with graph (∂Pe/∂ρ, ∂Pe/∂T, ∂Se/∂T)")
    print(f"{'='*80}")
    print(f"  {'N_cond':>{cw}}  {'Total (ms)':>14}  {'Fwd (ms)':>12}  "
          f"{'Post (ms)':>12}  {'Bwd (ms)':>12}  {'µs/cond':>10}")
    print("  " + "\u2500" * 78)
    for r in results:
        print(f"  {r['n_cond']:>{cw}}  "
              f"{r['total_mean']:>8.2f}±{r['total_std']:<5.2f}  "
              f"{r['fwd_mean']:>6.2f}±{r['fwd_std']:<5.2f}  "
              f"{r['post_mean']:>6.2f}±{r['post_std']:<5.2f}  "
              f"{r['bwd_mean']:>6.2f}±{r['bwd_std']:<5.2f}  "
              f"{r['total_pc_mean']*1e3:>10.4f}")
    print()

    # ------------------------------------------------------------------
    # Comparison table with no-graph results
    # ------------------------------------------------------------------
    if NOGRAPH_NPZ.exists():
        ng = np.load(NOGRAPH_NPZ)
        ng_dict = {int(nc): (float(pc), float(ps))
                   for nc, pc, ps in zip(ng["n_cond"],
                                         ng["total_pc_mean"],
                                         ng["total_pc_std"])}
        print(f"{'='*80}")
        print(f"Comparison: no-graph vs with-graph per-condition cost (µs)")
        print(f"{'='*80}")
        print(f"  {'N_cond':>{cw}}  {'No-graph (µs)':>16}  {'With-graph (µs)':>18}  {'Overhead':>10}")
        print("  " + "\u2500" * 60)
        for r in results:
            nc = r["n_cond"]
            wg = r["total_pc_mean"] * 1e3
            if nc in ng_dict:
                ng_v, _ = ng_dict[nc]
                ng_v_us = ng_v * 1e3
                overhead = (wg / ng_v_us - 1.0) * 100.0
                print(f"  {nc:>{cw}}  {ng_v_us:>16.4f}  {wg:>18.4f}  {overhead:>9.1f}%")
            else:
                print(f"  {nc:>{cw}}  {'—':>16}  {wg:>18.4f}  {'—':>10}")
        print()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    plot_dir = Path(__file__).resolve().parents[1] / "plots" / "helmholtz_graph_benchmark"
    plot_dir.mkdir(parents=True, exist_ok=True)

    arrays = {
        "n_cond":        np.array([r["n_cond"]       for r in results]),
        "total_mean":    np.array([r["total_mean"]    for r in results]),
        "total_std":     np.array([r["total_std"]     for r in results]),
        "fwd_mean":      np.array([r["fwd_mean"]      for r in results]),
        "fwd_std":       np.array([r["fwd_std"]       for r in results]),
        "post_mean":     np.array([r["post_mean"]     for r in results]),
        "post_std":      np.array([r["post_std"]      for r in results]),
        "bwd_mean":      np.array([r["bwd_mean"]      for r in results]),
        "bwd_std":       np.array([r["bwd_std"]       for r in results]),
        "total_pc_mean": np.array([r["total_pc_mean"] for r in results]),
        "total_pc_std":  np.array([r["total_pc_std"]  for r in results]),
    }
    np.savez(plot_dir / "helmholtz_graph_results.npz", **arrays)
    print(f"Results saved to {plot_dir}/helmholtz_graph_results.npz")

    _plot(arrays, plot_dir, device)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hydro EoS benchmark — batched forward + backward (derivatives)."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda", "auto"])
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = load_eos_config(args.config)
    device = args.device or "cuda"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    run_benchmark(cfg, device)


if __name__ == "__main__":
    main()
