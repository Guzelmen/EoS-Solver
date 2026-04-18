"""
Batched (ρ, T) Helmholtz pipeline — timing benchmark.

Runs 10×10 = 100 (ρ, T) conditions in a single batched forward pass and
measures wall time over N_TIMED runs (preceded by N_WARMUP discarded runs).

Batching strategy
-----------------
  - x grid: N_X log-spaced points, shared across all conditions.
  - Input tensor shape: [100 * N_X, 3] = [25600, 3] (one model(inputs_t) call).
  - phi output reshaped to [100, N_X].
  - All post-processing (xi, FD integrals, trapz, virial, Helmholtz) is
    vectorised over the condition dimension — no Python loop.

Timing
------
  - N_WARMUP runs discarded (JIT / cache warm-up).
  - N_TIMED runs measured.
  - Each run records:
      t_forward   : model forward pass
      t_postproc  : everything after forward (xi → F_physical)
      t_total     : t_forward + t_postproc
  - Reported as mean ± std across N_TIMED runs, and per-condition cost
    (divide by N_CONDITIONS = 100).

Usage
-----
    python -m scripts.helmholtz_batch_benchmark
    python -m scripts.helmholtz_batch_benchmark --config configs/helmholtz_pipeline.yaml --device cpu
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_loader import load_eos_config, load_pinn, build_x_grid
from src.inputs import (
    z_scale_inputs,
    compute_gamma,
    compute_lambda,
    r0_from_density,
    B_M,
    KEV_TO_J,
    M_PROTON,
)
from src.fd_integrals import fermi_dirac_half, fermi_dirac_three_half
from src.quantities.internal_energy import C_K, C_UEN, C_PRESSURE

# ---------------------------------------------------------------------------
# Benchmark conditions (edit here)
# ---------------------------------------------------------------------------

Z = 26.0   # Iron
A = 56.0

RHO_LIST = [54.62, 1.030, 0.9280, 11.455, 4.154, 11.874, 26.28, 0.2996, 0.7644, 0.8974]  # g/cm³
T_LIST   = [14.660, 0.9892, 0.5297, 0.4926, 0.2923, 0.2231, 0.2366, 0.3416, 0.2381, 0.1476]  # keV

# ---------------------------------------------------------------------------
# Benchmark parameters
# ---------------------------------------------------------------------------

N_WARMUP = 5
N_TIMED  = 20

# ---------------------------------------------------------------------------
# Timing helpers (shared with helmholtz_pipeline.py convention)
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
# Per-condition input preparation (returns numpy arrays, no torch yet)
# ---------------------------------------------------------------------------

def _build_condition_params(Z, A, rho_list, T_list):
    """
    For each (rho, T) pair compute (alpha_1, T_1, gamma, lam, r_0_1, V_1).
    Returns arrays of shape [N_CONDITIONS].
    """
    conditions = []
    for rho in rho_list:
        for T_keV in T_list:
            r0           = r0_from_density(rho, A)
            alpha_1, T_1 = z_scale_inputs(Z, r0, T_keV)
            gamma        = compute_gamma(T_1)
            lam          = compute_lambda(alpha_1, T_1)
            r_0_1        = alpha_1 * B_M
            V_1          = (4.0 / 3.0) * math.pi * r_0_1**3
            conditions.append((alpha_1, T_1, gamma, lam, r_0_1, V_1))

    alpha_1_arr = np.array([c[0] for c in conditions], dtype=np.float32)
    T_1_arr     = np.array([c[1] for c in conditions], dtype=np.float32)
    gamma_arr   = np.array([c[2] for c in conditions], dtype=np.float32)
    lam_arr     = np.array([c[3] for c in conditions], dtype=np.float32)
    r_0_1_arr   = np.array([c[4] for c in conditions], dtype=np.float64)
    V_1_arr     = np.array([c[5] for c in conditions], dtype=np.float64)
    return alpha_1_arr, T_1_arr, gamma_arr, lam_arr, r_0_1_arr, V_1_arr


# ---------------------------------------------------------------------------
# Single timed run
# ---------------------------------------------------------------------------

def _run_once(
    model,
    inputs_t: torch.Tensor,       # [N_COND * N_X, 3]  — pre-built, reused each run
    x_t: torch.Tensor,            # [N_X]
    gamma_t: torch.Tensor,        # [N_COND]
    lam_t: torch.Tensor,          # [N_COND]
    r_0_1_t: torch.Tensor,        # [N_COND]
    V_1_t: torch.Tensor,          # [N_COND]
    T_1_t: torch.Tensor,          # [N_COND]
    Z: float,
    A: float,
    device: str,
    n_cond: int,
    n_x: int,
) -> dict:
    """Execute one full pipeline pass and return timing (ms)."""

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    t0_fwd = _tick(device)

    # inputs_t already has requires_grad_=True (set once at build time)
    phi_flat = model(inputs_t)              # [N_COND * N_X, 1] or [N_COND * N_X]
    phi_t    = phi_flat.reshape(n_cond, n_x).detach()   # [N_COND, N_X]

    t_forward = _tock(t0_fwd, device)

    # ------------------------------------------------------------------
    # Post-processing (vectorised over condition dim)
    # ------------------------------------------------------------------
    t0_post = _tick(device)

    # xi: [N_COND, N_X]
    # gamma_t/lam_t: [N_COND] -> unsqueeze to [N_COND, 1] for broadcasting
    xi_t = gamma_t.unsqueeze(1) * phi_t / (lam_t.unsqueeze(1) * x_t.unsqueeze(0))

    # Boundary value xi_1: [N_COND]
    xi_1_t = xi_t[:, -1]

    # FD integrals on full grid: input shape [N_COND, N_X, 1] (batch dim expected)
    xi_col_t = xi_t.unsqueeze(2)                             # [N_COND, N_X, 1]
    F32_t    = fermi_dirac_three_half(xi_col_t).squeeze(2)   # [N_COND, N_X]
    F12_t    = fermi_dirac_half(xi_col_t).squeeze(2)         # [N_COND, N_X]

    # Boundary FD integral: [N_COND, 1, 1] -> [N_COND]
    xi_1_col_t = xi_1_t.unsqueeze(1).unsqueeze(2)            # [N_COND, 1, 1]
    F32_b_t    = fermi_dirac_three_half(xi_1_col_t).squeeze(2).squeeze(1)  # [N_COND]

    # Volume integrals using torch.trapezoid over x dimension
    # K_1[c] = C_K * r_0_1[c]^3 * T_1[c]^2.5 * trapz(F32[c,:] * x^2, x)
    integrand_K   = F32_t * x_t.unsqueeze(0)**2             # [N_COND, N_X]
    integrand_Uen = F12_t * x_t.unsqueeze(0)                # [N_COND, N_X]

    trap_K   = torch.trapezoid(integrand_K,   x_t, dim=1)   # [N_COND]
    trap_Uen = torch.trapezoid(integrand_Uen, x_t, dim=1)   # [N_COND]

    K_1_t   = C_K   * r_0_1_t**3 * T_1_t**2.5 * trap_K    # [N_COND]
    U_en_1_t = -C_UEN * r_0_1_t**2 * T_1_t**1.5 * trap_Uen # [N_COND]

    # Pressure and virial: U_ee = 3*P*V - 2*K - U_en
    P_e1_t  = C_PRESSURE * T_1_t**2.5 * F32_b_t             # [N_COND]
    U_ee_1_t = 3.0 * P_e1_t * V_1_t - 2.0 * K_1_t - U_en_1_t  # [N_COND]

    # Chemical potential: mu_e1 = xi_1 * T_1 * KEV_TO_J
    mu_e1_t = xi_1_t * T_1_t * KEV_TO_J                     # [N_COND] [J]

    # Helmholtz (Z=1): F_Z1 = mu_e1 - 2/3*K_1 - U_ee_1
    F_Z1_t = mu_e1_t - (2.0 / 3.0) * K_1_t - U_ee_1_t      # [N_COND]

    # Z-scale to physical: F_phys = Z^(7/3) * F_Z1
    F_phys_t = Z**(7.0 / 3.0) * F_Z1_t                      # [N_COND] [J/atom]

    # Unit conversions
    F_J_kg_t  = F_phys_t / (A * M_PROTON)                   # [N_COND] [J/kg]
    F_erg_g_t = F_J_kg_t * 1e4                              # [N_COND] [erg/g]

    t_postproc = _tock(t0_post, device)

    return {
        "t_forward":   t_forward,
        "t_postproc":  t_postproc,
        "t_total":     t_forward + t_postproc,
        # Representative scalar outputs (last condition) for sanity check
        "F_erg_g_last": float(F_erg_g_t[-1].item()),
        "xi_1_last":    float(xi_1_t[-1].item()),
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(cfg: dict, device_str: str):
    n_x   = cfg.get("n_x",   256)
    x_min = cfg.get("x_min", 1e-6)

    print(f"\n{'='*60}")
    print(f"Batched Helmholtz benchmark")
    print(f"  Z={int(Z)}, A={int(A)}")
    print(f"  rho conditions : {RHO_LIST}")
    print(f"  T conditions   : {T_LIST}")
    print(f"  N_conditions   : {len(RHO_LIST) * len(T_LIST)}")
    print(f"  N_X per cond   : {n_x}")
    print(f"  Input tensor   : [{len(RHO_LIST)*len(T_LIST)*n_x}, 3]")
    print(f"  Device         : {device_str}")
    print(f"  Warmup runs    : {N_WARMUP}")
    print(f"  Timed runs     : {N_TIMED}")
    print(f"{'='*60}\n")

    # ---------------------------------------------------------------
    # Load model once
    # ---------------------------------------------------------------
    model, _params = load_pinn(cfg, device=device_str, verbose=True)
    model.eval()
    torch.set_grad_enabled(True)

    n_cond = len(RHO_LIST) * len(T_LIST)

    # ---------------------------------------------------------------
    # Build condition parameter arrays (CPU, reused each run)
    # ---------------------------------------------------------------
    alpha_1_arr, T_1_arr, gamma_arr, lam_arr, r_0_1_arr, V_1_arr = (
        _build_condition_params(Z, A, RHO_LIST, T_LIST)
    )

    # ---------------------------------------------------------------
    # Build x grid and broadcast to [N_COND * N_X, 3] input tensor
    # ---------------------------------------------------------------
    x_np = build_x_grid(n_x, x_min)   # [N_X]

    # Repeat x for each condition: tile x N_COND times
    x_rep = np.tile(x_np.astype(np.float32), n_cond)                          # [N_COND * N_X]
    # Repeat each (alpha_1, T_1) pair N_X times
    alpha_rep = np.repeat(alpha_1_arr, n_x)                                    # [N_COND * N_X]
    T_rep     = np.repeat(T_1_arr,     n_x)                                    # [N_COND * N_X]

    inputs_np = np.stack([x_rep, alpha_rep, T_rep], axis=1)                   # [N_COND*N_X, 3]
    inputs_t  = (
        torch.from_numpy(inputs_np).to(device_str).requires_grad_(True)
    )

    # Move condition-level arrays to device as float32 tensors
    def _to_t(arr, dtype=torch.float32):
        return torch.from_numpy(arr.astype(np.float32)).to(device_str)

    x_t     = torch.from_numpy(x_np.astype(np.float32)).to(device_str)
    gamma_t  = _to_t(gamma_arr)
    lam_t    = _to_t(lam_arr)
    r_0_1_t  = _to_t(r_0_1_arr)
    V_1_t    = _to_t(V_1_arr)
    T_1_t    = _to_t(T_1_arr)

    # ---------------------------------------------------------------
    # Warmup
    # ---------------------------------------------------------------
    print(f"Running {N_WARMUP} warmup passes...")
    for _ in range(N_WARMUP):
        _run_once(
            model, inputs_t, x_t, gamma_t, lam_t, r_0_1_t, V_1_t, T_1_t,
            Z, A, device_str, n_cond, n_x,
        )
    print("Warmup done.\n")

    # ---------------------------------------------------------------
    # Timed runs
    # ---------------------------------------------------------------
    records = []
    print(f"Running {N_TIMED} timed passes...")
    for i in range(N_TIMED):
        rec = _run_once(
            model, inputs_t, x_t, gamma_t, lam_t, r_0_1_t, V_1_t, T_1_t,
            Z, A, device_str, n_cond, n_x,
        )
        records.append(rec)
        print(f"  run {i+1:2d}/{N_TIMED}: total={rec['t_total']:7.2f} ms  "
              f"fwd={rec['t_forward']:6.2f} ms  post={rec['t_postproc']:6.2f} ms  "
              f"F_last={rec['F_erg_g_last']:.4e} erg/g")

    # ---------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------
    t_fwd   = np.array([r["t_forward"]  for r in records])
    t_post  = np.array([r["t_postproc"] for r in records])
    t_total = np.array([r["t_total"]    for r in records])

    # Per-condition cost
    t_fwd_pc   = t_fwd   / n_cond
    t_post_pc  = t_post  / n_cond
    t_total_pc = t_total / n_cond

    col_w = 36

    print(f"\n{'='*60}")
    print(f"Results — {n_cond} conditions, {n_x} x-points each")
    print(f"Device: {device_str}\n")

    header = f"  {'Stage':<{col_w}}{'Mean (ms)':>12}{'Std (ms)':>12}{'Mean/cond (ms)':>16}{'Std/cond (ms)':>14}"
    sep    = "  " + "\u2500" * (col_w + 54)
    print(header)
    print(sep)

    rows = [
        ("Forward pass",   t_fwd,   t_fwd_pc),
        ("Post-processing", t_post, t_post_pc),
        ("Total",          t_total, t_total_pc),
    ]
    for name, arr, arr_pc in rows:
        print(f"  {name:<{col_w}}"
              f"{arr.mean():>12.3f}"
              f"{arr.std():>12.3f}"
              f"{arr_pc.mean():>16.4f}"
              f"{arr_pc.std():>14.4f}")

    print(sep)
    print(f"\n  Sanity check (last condition: rho={RHO_LIST[-1]} g/cm³, T={T_LIST[-1]} keV):")
    print(f"    F_electron = {records[-1]['F_erg_g_last']:.6e} erg/g")
    print(f"    xi_1       = {records[-1]['xi_1_last']:.6g}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batched Helmholtz free energy pipeline — timing benchmark."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to EoS config yaml (default: configs/default.yaml)")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda", "auto"],
                        help="Device override (default: cpu)")
    return parser.parse_args()


def main():
    args   = parse_args()
    cfg    = load_eos_config(args.config)
    device = args.device or "cpu"
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    run_benchmark(cfg, device)


if __name__ == "__main__":
    main()
