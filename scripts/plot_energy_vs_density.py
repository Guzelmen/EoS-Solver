"""
Plot electron internal energy (erg/g) vs mass density for Aluminium (Z=13, A=27)
at three temperatures: T = 1, 10, 100 keV.

Usage (from EoS-Solver root):
    python -m scripts.plot_e_vs_density

Pipeline per (rho, T) point:
    rho -> r0 -> (alpha_1, T_1) -> gamma, lam -> phi(x) on log grid
    -> xi(x) on full grid -> K_1, U_en_1, U_ee_1 (virial) -> E_e [erg/g]
"""

import argparse
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model_loader import load_pinn, predict_phi, build_x_grid
from src.inputs import (
    z_scale_inputs, compute_gamma, compute_lambda,
    compute_xi, r0_from_density,
)
from src.quantities.internal_energy import compute_total_energy

# ---------------------------------------------------------------------------
# Element parameters
# ---------------------------------------------------------------------------
Z_AL = 13
A_AL = 27

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
N_RHO   = 60
RHO_MIN = 0.1    # g/cm^3
RHO_MAX = 10     # g/cm^3

N_X     = 1024
X_MIN   = 1e-6    # log-spaced, clusters points near origin for integral accuracy

TEMPS_KEV = [1.0, 10.0, 100.0]
COLORS    = ["steelblue", "darkorange", "forestgreen"]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="E vs density plot for aluminium.")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to EoS config yaml. Defaults to configs/default.yaml.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for the figure. Defaults to plots/<run_name>/energy/...")
    args = parser.parse_args()

    # Load config and model
    cfg_path = Path(args.config) if args.config else REPO_ROOT / "configs" / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    print("--- Config ---")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("--------------")

    # Build output path from run_name / epoch if not explicitly provided
    if args.output is None:
        run_name = cfg.get("run_name", "default")
        epoch    = cfg.get("epoch", None)
        tag      = f"{run_name}_epoch_{epoch}" if epoch is not None else run_name
        out_path = REPO_ROOT / "plots" / tag / "energy" / "plot_e_erg_g_vs_density_al.png"
    else:
        out_path = Path(args.output)

    device = cfg.get("device", "cpu")
    model, params = load_pinn(cfg, device)

    # Grids
    rho_grid = np.logspace(np.log10(RHO_MIN), np.log10(RHO_MAX), N_RHO)
    x_grid   = np.logspace(np.log10(X_MIN), 0.0, N_X)   # log-spaced in x

    fig, ax = plt.subplots(figsize=(8, 6))

    for T_phys, color in zip(TEMPS_KEV, COLORS):
        energies = []

        for rho in rho_grid:
            # Physical -> Z=1 reduced variables
            r0          = r0_from_density(rho, A_AL)
            alpha_1, T_1 = z_scale_inputs(Z_AL, r0, T_phys)

            # Derived constants for FD argument
            gamma = compute_gamma(T_1)
            lam   = compute_lambda(alpha_1, T_1)

            # PINN inference -> phi on full log grid
            phi = predict_phi(alpha_1, T_1, x_grid, model, device)

            # FD argument xi(x) on full grid
            xi      = compute_xi(phi, x_grid, gamma, lam)
            xi_1    = float(xi[-1])    # boundary value at x=1

            # Total electron energy [erg/g]
            result = compute_total_energy(
                xi      = xi,
                x       = x_grid,
                xi_1    = xi_1,
                T_1_keV = T_1,
                alpha_1 = alpha_1,
                Z       = Z_AL,
                A       = A_AL,
            )
            energies.append(result["E_e_erg_g"])

        energies = np.array(energies)

        # Mask non-positive values before log plot (can occur when virial
        # cancellation gives negative net energy at extreme low density/OOD)
        valid = energies > 0
        ax.plot(
            rho_grid[valid],
            energies[valid],
            color=color,
            linewidth=1.8,
            label=f"T = {T_phys:.0f} keV",
        )
        if not valid.all():
            n_bad = (~valid).sum()
            print(f"  T={T_phys} keV: {n_bad} non-positive energy points masked")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Density $\rho$ (g/cm$^3$)", fontsize=12)
    ax.set_ylabel(r"Electron Energy $E_e$ (erg/g)", fontsize=12)
    ax.set_title(f"Thomas-Fermi Electron EoS — Aluminium (Z={Z_AL}, A={A_AL})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()