"""
P vs density plot for aluminium across 4 temperatures.

Produces a single figure with 4 lines (one per temperature) on log-log axes,
sweeping over density in [0.1, 10] g/cm^3 and computing electron pressure
via the PINN + Z-scaling pipeline.

Usage:
    python -m scripts.plot_P_vs_density
    python -m scripts.plot_P_vs_density --config configs/default.yaml
    python -m scripts.plot_P_vs_density --config configs/default.yaml --output plots/P_vs_rho_Al.png

Element: Aluminium  Z=13, A=27
Temperatures: 0.01, 0.1, 1, 10 keV

IMPORTANT — T range warning:
    For Al Z=13, Z^(4/3) ~ 30.56. The PINN operates in the Z=1 reduced system,
    so T_1 = T_phys / 30.56:
        T=0.01 keV  ->  T_1 ~ 3.3e-4 keV
        T=0.1  keV  ->  T_1 ~ 3.3e-3 keV
        T=1.0  keV  ->  T_1 ~ 0.033  keV
        T=10   keV  ->  T_1 ~ 0.33   keV
    The PINN training data covered T values of order 1-10 keV in Z=1 space.
    The two lowest physical temperatures will likely fall far outside the
    training distribution and should be treated with caution until validated
    against a numerical solver.

Note on A:
    A=27 appears only in the density <-> r0 conversion (x-axis).
    The pressure formula contains no A dependence — it is a purely
    electronic quantity that Z-scales independently of ion mass.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make sure the EoS repo src/ is importable when called as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_loader  import load_eos_config, load_pinn, build_x_grid, predict_phi
from src.inputs        import z_scale_inputs, compute_gamma, compute_lambda, \
                              r0_from_density, density_from_r0
from src.quantities.pressure import compute_pressure, pa_to_mbar


# ---------------------------------------------------------------------------
# Element parameters
# ---------------------------------------------------------------------------
Z_AL = 13
A_AL = 27   # atomic mass number — only used for density <-> r0 conversion


# ---------------------------------------------------------------------------
# Sweep parameters
# ---------------------------------------------------------------------------
T_VALUES_KEV  = [1, 10, 100]    # physical temperatures [keV]
N_RHO         = 60                          # number of density points
RHO_MIN       = 0.1                         # [g/cm^3]
RHO_MAX       = 10.0                        # [g/cm^3]

# Colour cycle for the 4 temperature lines
COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="P vs density plot for aluminium, 4 temperatures."
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to EoS config yaml. Defaults to configs/default.yaml.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for the figure. Defaults to plots/<run_name>/pressure/P_vs_rho_Al.png.",
    )
    parser.add_argument(
        "--n_rho", type=int, default=N_RHO,
        help="Number of density grid points.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------ setup
    cfg = load_eos_config(args.config)
    print("--- Config ---")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("--------------")
    n_x   = cfg.get("n_x",   512)
    x_min = cfg.get("x_min", 1e-4)

    # Build output path from run_name / epoch if not explicitly provided
    if args.output is None:
        run_name = cfg.get("run_name", "default")
        epoch    = cfg.get("epoch", None)
        tag      = f"{run_name}_epoch_{epoch}" if epoch is not None else run_name
        out_path = Path("plots") / tag / "pressure" / "P_vs_rho_Al.png"
    else:
        out_path = Path(args.output)

    print("Loading PINN...")
    model, _ = load_pinn(cfg)
    device   = cfg.get("device", "cpu")

    # Log-spaced density grid in physical space [g/cm^3]
    rho_grid = np.logspace(np.log10(RHO_MIN), np.log10(RHO_MAX), args.n_rho)

    # x grid is the same for every (rho, T) point
    x_grid = build_x_grid(n_x, x_min)

    # only need x at boundary
    x_boundary = np.array([1.0])

    # --------------------------------------------------------- compute curves
    results = {}   # T_keV -> (rho_arr, P_mbar_arr)

    for T_phys in T_VALUES_KEV:
        print(f"\nT = {T_phys} keV  (T_1 = {T_phys / Z_AL**(4/3):.4g} keV in Z=1 system)")

        P_mbar_list = []
        rho_list    = []

        for rho in rho_grid:
            # 1. density -> physical cell radius
            r0 = r0_from_density(rho, A_AL)

            # 2. Z-scale to Z=1 reduced system
            alpha_1, T_1 = z_scale_inputs(Z_AL, r0, T_phys)

            # 3. Derived FMT constants (Z=1 system)
            gamma = compute_gamma(T_1)
            lam   = compute_lambda(alpha_1, T_1)

            # 4. PINN forward pass -> phi(x) on grid
            phi = predict_phi(alpha_1, T_1, x_boundary, model, device)

            # phi[-1] is phi at x=1 (boundary); this is what enters pressure
            phi_boundary = (float(phi[-1]) if len(phi) == 1 else None)


            if phi_boundary < 0:
                print(f"  [info] phi(1) = {phi_boundary:.4g} < 0 at rho={rho:.3g} g/cm^3 "
                      f"(xi_1={gamma*phi_boundary/lam:.4g}, classical regime)")

            # 5. Electron pressure [Pa] -> [Mbar]
            result  = compute_pressure(phi_boundary, T_1, gamma, lam, Z=Z_AL)
            P_mbar  = pa_to_mbar(result["P_e"])

            rho_list.append(rho)
            P_mbar_list.append(P_mbar)

        results[T_phys] = (np.array(rho_list), np.array(P_mbar_list))
        print(f"  P range: [{P_mbar_list[0]:.3e}, {P_mbar_list[-1]:.3e}] Mbar")

    # ------------------------------------------------------------------- plot
    fig, ax = plt.subplots(figsize=(7, 5))

    for (T_phys, colour) in zip(T_VALUES_KEV, COLOURS):
        rho_arr, P_arr = results[T_phys]
        # Mask non-positive pressures before log plot
        mask = P_arr > 0
        ax.plot(
            rho_arr[mask], P_arr[mask],
            color=colour,
            linewidth=1.8,
            label=f"T = {T_phys} keV",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Density $\rho$ (g/cm$^3$)", fontsize=12)
    ax.set_ylabel(r"Electron Pressure $P_e$ (Mbar)", fontsize=12)
    ax.set_title("Thomas-Fermi Electron EoS — Aluminium (Z=13, A=27)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    print(f"\nSaved figure: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()