"""
End-to-end electron Helmholtz free energy pipeline with per-stage timing.

Runs the full F_e pipeline from physical inputs (Z, A, rho, T) to F_physical,
preserving the autograd computation graph from network inputs through to the
final result so that torch.autograd.grad(F_physical, ...) works downstream.

Usage:
    python -m scripts.helmholtz_pipeline \\
        --Z 26 --A 56 --rho 7.87 --T 1.0

    python -m scripts.helmholtz_pipeline \\
        --Z 26 --A 56 --rho 7.87 --T 1.0 \\
        --config configs/default.yaml \\
        --device cuda

Pipeline stages (timed individually):
  1. Input transform / Z-scaling
  2. Build x grid
  3. Model load
  4. Network forward pass — phi(x)
  5. FD argument xi(x)
  6. Boundary-only quantities (mu, P, Q)
  7. Volume integrals (K, U_en)
  8. Virial theorem — U_ee
  9. Helmholtz F (Z=1)
  10. Z-scale output
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
)
from src.fd_integrals import fermi_dirac_half, fermi_dirac_three_half
from src.quantities.internal_energy import C_K, C_UEN, C_PRESSURE
from src.quantities.chemical_potential import compute_chemical_potential
from src.quantities.helmholtz import compute_helmholtz

# ---------------------------------------------------------------------------
# Grid defaults (match repo convention)
# ---------------------------------------------------------------------------
N_X   = 1024
X_MIN = 1e-6

# OOD warning thresholds
T_1_MIN    = 0.01
T_1_MAX    = 10.0
ALPHA_1_MIN = 1.0
ALPHA_1_MAX = 10.0


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _sync(device: str):
    """CUDA barrier before timing checkpoints; no-op on CPU."""
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _tick(device: str) -> float:
    _sync(device)
    return time.perf_counter()


def _tock(t0: float, device: str) -> float:
    _sync(device)
    return (time.perf_counter() - t0) * 1e3   # ms


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    Z: float,
    A: float,
    rho_gcc: float,
    T_keV: float,
    cfg: dict,
    device_override: str = None,
) -> dict:
    """
    Full Helmholtz free energy pipeline.

    Returns a dict with all intermediates and a 'timing' sub-dict (all in ms).

    Graph behaviour is controlled by cfg['use_graph'] (default False):
      use_graph=True  — full autograd graph from phi(x) through to F_phys_t is
                        preserved; torch.autograd.grad(F_phys_t, inputs_t) works.
      use_graph=False — detach / no_grad applied at every opportunity for speed.
    """
    # Resolve device and graph mode
    # Device comes from CLI (device_override); config no longer carries a device key.
    target_device = device_override or "cpu"
    if target_device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    use_graph = bool(cfg.get("use_graph", False))

    # Grad must stay enabled for the model forward pass — the BC hard-constraint
    # transform in the PINN calls first_deriv_auto (autograd) internally, so
    # torch.no_grad() would break inference regardless of use_graph.
    # We control graph retention by detaching phi after the forward pass instead.
    torch.set_grad_enabled(True)

    timing = {}

    # ------------------------------------------------------------------
    # Stage 1: Input transform / Z-scaling
    # ------------------------------------------------------------------
    t0 = _tick(target_device)
    r0              = r0_from_density(rho_gcc, A)
    alpha_1, T_1    = z_scale_inputs(Z, r0, T_keV)
    gamma           = compute_gamma(T_1)
    lam             = compute_lambda(alpha_1, T_1)
    r_0_1           = alpha_1 * B_M                       # [m]
    V_1             = (4.0 / 3.0) * math.pi * r_0_1**3   # cell volume [m^3]
    timing["Input transform / Z-scaling"] = _tock(t0, target_device)

    # OOD warnings
    if not (T_1_MIN <= T_1 <= T_1_MAX):
        print(f"  [warn] T_1 = {T_1:.4g} keV — likely OOD (training range {T_1_MIN}-{T_1_MAX} keV)")
    if not (ALPHA_1_MIN <= alpha_1 <= ALPHA_1_MAX):
        print(f"  [warn] alpha_1 = {alpha_1:.4g} — likely OOD (training range {ALPHA_1_MIN}-{ALPHA_1_MAX})")

    # ------------------------------------------------------------------
    # Stage 2: Build x grid
    # ------------------------------------------------------------------
    t0 = _tick(target_device)
    x_np  = build_x_grid(N_X, X_MIN)                      # [N_X] numpy
    x_t   = torch.tensor(x_np, dtype=torch.float32, device=target_device)
    timing["Build x grid"] = _tock(t0, target_device)

    # ------------------------------------------------------------------
    # Stage 3: Model load
    # ------------------------------------------------------------------
    t0 = _tick(target_device)
    model, _params = load_pinn(cfg, device=target_device, verbose=True)
    model.eval()

    timing["Model load"] = _tock(t0, target_device)

    # ------------------------------------------------------------------
    # Stage 4: Network forward pass — phi(x)
    # ------------------------------------------------------------------
    t0 = _tick(target_device)
    n      = len(x_np)
    alpha_col = np.full(n, alpha_1, dtype=np.float32)
    T_col     = np.full(n, T_1,     dtype=np.float32)
    inputs_np = np.stack([x_np.astype(np.float32), alpha_col, T_col], axis=1)

    # inputs_t always needs requires_grad=True — the model's forward pass calls
    # first_deriv_auto internally (hard BC transform), which requires a grad-enabled input.
    inputs_t = torch.from_numpy(inputs_np).to(target_device).requires_grad_(True)

    phi_t = model(inputs_t)              # shape [N_X, 1] or [N_X]
    phi_t = phi_t.reshape(-1)            # [N_X]
    if not use_graph:
        # Detach after forward: graph not needed downstream, free the memory.
        phi_t = phi_t.detach()
    timing["Network forward (phi)"] = _tock(t0, target_device)

    # ------------------------------------------------------------------
    # Stage 5: FD argument xi(x)
    # ------------------------------------------------------------------
    t0 = _tick(target_device)
    xi_t  = gamma * phi_t / (lam * x_t)  # [N_X]
    # use_graph=True: keep xi_1 as a tensor so the boundary->mu/P path stays in-graph
    # use_graph=False: scalar float is fine
    xi_1_t = xi_t[-1].unsqueeze(0).unsqueeze(0)   # [1, 1], always a tensor
    xi_1   = float(xi_t[-1].item())               # scalar for reporting
    timing["FD argument xi(x)"] = _tock(t0, target_device)

    # ------------------------------------------------------------------
    # Stage 6: Boundary-only quantities (mu, P, Q)
    #
    # use_graph=True:
    #   xi_1_t carries the graph from phi_t; mu_e1_t and P_e1_t are tensors
    #   so that the mu and P*V terms in F and U_ee are differentiable.
    # use_graph=False:
    #   everything collapses to Python floats via compute_chemical_potential.
    # ------------------------------------------------------------------
    t0 = _tick(target_device)

    if use_graph:
        # mu_e1 = xi_1 * T_1_keV * KEV_TO_J   (inline, keeps graph through xi_1_t)
        mu_e1_t = xi_1_t * T_1 * KEV_TO_J          # [1, 1] tensor [J]
        mu_e1   = float(mu_e1_t.item())

        # P_e1 = C_PRESSURE * T_1^(5/2) * F_{3/2}(xi_1_t)  (in-graph)
        F32_b_t = fermi_dirac_three_half(xi_1_t)    # [1, 1] tensor
        F32_b   = float(F32_b_t.item())
        P_e1_t  = C_PRESSURE * T_1**2.5 * F32_b_t  # [1, 1] tensor [Pa]
        P_e1    = float(P_e1_t.item())

    else:
        phi_boundary = float(phi_t[-1].item())
        mu_result    = compute_chemical_potential(
            phi_boundary=phi_boundary,
            T_1_keV=T_1,
            gamma=gamma,
            lam=lam,
            Z=Z,
        )
        mu_e1  = mu_result["mu_e1"]                 # float [J]
        mu_e1_t = None

        xi_1_scalar_t = torch.tensor([[xi_1]], dtype=torch.float32, device=target_device)
        F32_b  = float(fermi_dirac_three_half(xi_1_scalar_t).item())
        P_e1   = C_PRESSURE * T_1**2.5 * F32_b     # float [Pa]
        P_e1_t = None

    P_e = Z**(10.0 / 3.0) * P_e1                   # float [Pa], always

    timing["Boundary-only quantities"] = _tock(t0, target_device)

    # ------------------------------------------------------------------
    # Stage 7: Volume integrals — K and U_en
    # Keep in-graph by using fermi_dirac on the xi tensor and torch.trapezoid.
    # ------------------------------------------------------------------
    t0 = _tick(target_device)

    # Reshape xi_t to [N_X, 1] as FD functions expect batched input
    xi_col_t = xi_t.unsqueeze(1)                          # [N_X, 1]

    F32_t    = fermi_dirac_three_half(xi_col_t).reshape(-1)  # [N_X]
    F12_t    = fermi_dirac_half(xi_col_t).reshape(-1)        # [N_X]

    # K_1 = C_K * r_0_1^3 * T_1^(5/2) * trapz(F_{3/2}(xi) * x^2, x)
    K_1_t    = C_K * r_0_1**3 * T_1**2.5 * torch.trapezoid(F32_t * x_t**2, x_t)

    # U_en_1 = -C_UEN * r_0_1^2 * T_1^(3/2) * trapz(F_{1/2}(xi) * x, x)
    U_en_1_t = -C_UEN * r_0_1**2 * T_1**1.5 * torch.trapezoid(F12_t * x_t, x_t)

    # Scalar floats for intermediate reporting
    K_1    = float(K_1_t.item())
    U_en_1 = float(U_en_1_t.item())
    timing["Volume integrals (K, Uen)"] = _tock(t0, target_device)

    # ------------------------------------------------------------------
    # Stage 8: Virial theorem — U_ee
    # U_ee_1 = 3*P_1*V_1 - 2*K_1 - U_en_1
    #
    # use_graph=True:  P_e1_t is a tensor (graph intact through xi_1_t -> phi_t)
    # use_graph=False: P_e1 is a float, product is a constant w.r.t. the graph
    # ------------------------------------------------------------------
    t0 = _tick(target_device)
    pv_term  = 3.0 * (P_e1_t.squeeze() if use_graph else P_e1) * V_1
    U_ee_1_t = pv_term - 2.0 * K_1_t - U_en_1_t
    U_ee_1   = float(U_ee_1_t.item())
    timing["Virial theorem (Uee)"] = _tock(t0, target_device)

    # ------------------------------------------------------------------
    # Stage 9: Helmholtz F (Z=1)
    # F_Z1 = mu_e1 - 2/3 * K_1 - U_ee_1   (all [J])
    # ------------------------------------------------------------------
    t0 = _tick(target_device)
    helm_result = compute_helmholtz(
        K_1=K_1,
        U_ee_1=U_ee_1,
        mu_e1=mu_e1,
        Z=Z,
        A=A,
    )
    F_Z1    = helm_result["F_Z1"]
    F_J_kg  = helm_result["F_J_kg"]
    F_erg_g = helm_result["F_erg_g"]
    timing["Helmholtz F (Z=1)"] = _tock(t0, target_device)

    # ------------------------------------------------------------------
    # Stage 10: Z-scale output
    #
    # use_graph=True:  all three terms are tensors -> full graph to phi_t
    # use_graph=False: mu_e1 is a float scalar; result is still a tensor
    #                  (K_1_t and U_ee_1_t are detached)
    # ------------------------------------------------------------------
    t0 = _tick(target_device)
    mu_term  = mu_e1_t.squeeze() if use_graph else mu_e1
    F_phys_t = Z**(7.0 / 3.0) * (
        mu_term  -  (2.0 / 3.0) * K_1_t  -  U_ee_1_t
    )                                                       # [J] per atom
    F_physical = float(F_phys_t.item())
    timing["Z-scale output"] = _tock(t0, target_device)

    return {
        # Inputs / reduced vars
        "r0":      r0,
        "alpha_1": alpha_1,
        "T_1":     T_1,
        "gamma":   gamma,
        "lam":     lam,
        "V_1":     V_1,
        # Intermediates (scalars)
        "xi_1":    xi_1,
        "mu_e1":   mu_e1,
        "mu_e":    Z**(4.0 / 3.0) * mu_e1,
        "P_e1":    P_e1,
        "P_e":     P_e,
        "F32_b":   F32_b,
        "K_1":     K_1,
        "U_en_1":  U_en_1,
        "U_ee_1":  U_ee_1,
        # Helmholtz results
        "F_Z1":    F_Z1,
        "F_J_kg":  F_J_kg,
        "F_erg_g": F_erg_g,
        "F_physical": F_physical,
        # Tensors for downstream autograd (only meaningful when use_graph=True)
        "inputs_t":  inputs_t,
        "F_phys_t":  F_phys_t,
        # Metadata
        "device":    target_device,
        "use_graph": use_graph,
        "timing":    timing,
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="TF-EoS electron Helmholtz free energy pipeline with timing."
    )
    parser.add_argument("--Z",      type=float, required=True,
                        help="Atomic number (e.g. 26 for Fe)")
    parser.add_argument("--A",      type=float, required=True,
                        help="Atomic mass number (e.g. 56 for Fe56)")
    parser.add_argument("--rho",    type=float, required=True,
                        help="Mass density [g/cm^3]")
    parser.add_argument("--T",      type=float, required=True,
                        help="Temperature [keV]")
    parser.add_argument("--config", type=str,   default=None,
                        help="Path to EoS config yaml (default: configs/default.yaml)")
    parser.add_argument("--device", type=str,   default=None,
                        choices=["cpu", "cuda", "auto"],
                        help="Device (default: from config)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    cfg    = load_eos_config(args.config)
    device = args.device or "cpu"

    print(f"\nRunning Helmholtz pipeline: Z={int(args.Z)}, A={int(args.A)}, "
          f"rho={args.rho} g/cm3, T={args.T} keV, device={device}")
    print(f"\n=== Config ===")
    print(f"  wandb_run_path : {cfg.get('wandb_run_path', '')}")
    print(f"  run_name       : {cfg.get('run_name', '')}")
    print(f"  epoch          : {cfg.get('epoch', 'null (latest)')}")
    print(f"  n_x            : {cfg.get('n_x', 'default')}")
    print(f"  x_min          : {cfg.get('x_min', 'default')}")
    print(f"  use_graph      : {cfg.get('use_graph', False)}")
    print()

    result = run_pipeline(
        Z=args.Z,
        A=args.A,
        rho_gcc=args.rho,
        T_keV=args.T,
        cfg=cfg,
        device_override=device,
    )

    timing = result["timing"]
    col_w  = 38

    # Build timing table
    sep = "\u2500" * (col_w + 14)
    stage_order = [
        "Input transform / Z-scaling",
        "Build x grid",
        "Model load",
        "Network forward (phi)",
        "FD argument xi(x)",
        "Boundary-only quantities",
        "Volume integrals (K, Uen)",
        "Virial theorem (Uee)",
        "Helmholtz F (Z=1)",
        "Z-scale output",
    ]

    pipeline_stages = [s for s in stage_order if s != "Model load"]
    t_model   = timing.get("Model load", 0.0)
    t_pipeline = sum(timing.get(s, 0.0) for s in pipeline_stages)
    t_total    = t_model + t_pipeline

    print(f"=== Helmholtz Pipeline Timing ===")
    print(f"Device: {result['device']}  |  use_graph: {result['use_graph']}\n")
    print(f"  {'Stage':<{col_w}}{'Time (ms)':>10}")
    print(f"  {sep}")
    for stage in stage_order:
        t_ms = timing.get(stage, 0.0)
        print(f"  {stage:<{col_w}}{t_ms:>10.2f}")
    print(f"  {sep}")
    print(f"  {'TOTAL (excl. model load)':<{col_w}}{t_pipeline:>10.2f}")
    print(f"  {'TOTAL (incl. model load)':<{col_w}}{t_total:>10.2f}")

    print(f"\n=== Intermediates (Z=1 system) ===")
    print(f"  alpha_1      = {result['alpha_1']:.6g}   [dimensionless]")
    print(f"  T_1          = {result['T_1']:.6g}   [keV]")
    print(f"  xi_1 (mu/kT) = {result['xi_1']:.6g}   [dimensionless]")
    print(f"  K_1          = {result['K_1']:.6g}   [J]")
    print(f"  U_en_1       = {result['U_en_1']:.6g}   [J]")
    print(f"  U_ee_1       = {result['U_ee_1']:.6g}   [J]")
    print(f"  mu_e1        = {result['mu_e1']:.6g}   [J]")
    print(f"  P_e1         = {result['P_e1']:.6g}   [Pa]")
    print(f"  F_Z1         = {result['F_Z1']:.6g}   [J/atom, Z=1]")

    print(f"\n=== Result ===")
    print(f"  F_electron   = {result['F_physical']:.6e}   [J/atom]")
    print(f"  F_electron   = {result['F_J_kg']:.6e}   [J/kg]")
    print(f"  F_electron   = {result['F_erg_g']:.6e}   [erg/g]")
    print(f"  P_electron   = {result['P_e'] / 1e11:.6g}   [Mbar]")
    print()


if __name__ == "__main__":
    main()
