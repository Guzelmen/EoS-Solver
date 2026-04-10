"""
Dimensionless TF-EoS calculator.

Computes Pv/kTZ, E_kin/kTZ, E_pot/kTZ for a list of (rho, T) pairs
and prints a formatted table to terminal.

E_pot/kTZ is obtained via the virial theorem (FMT Section VI):
    2*K + U_pot = 3*P*V  =>  U_pot/kTZ = (3*P_1*V_1 - 2*K_1) / (T_1*kev_to_J)

Usage examples:
    # Single point
    python -m scripts.eos_dimensionless \\
        --Z 26 --A 56 \\
        --rho 54.62 \\
        --T 14.660

    # Multiple points (paired by index, same length required)
    python -m scripts.eos_dimensionless \\
        --Z 26 --A 56 \\
        --rho 11.874 11.455 0.8974 0.7644 0.9280 26.28 54.62 4.154 1.030 0.2996 0.5235 \\
        --T 0.2231 0.4926 0.1476 0.2381 0.5297 0.2366 14.660 0.2923 0.9892 0.3416 0.0326

    # Optional: override config path
    python -m scripts.eos_dimensionless --Z 13 --A 27 --rho 1.0 --T 1.0 \\
        --config configs/default.yaml
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_loader import load_eos_config, load_pinn, build_x_grid, predict_phi
from src.inputs import z_scale_inputs, compute_gamma, compute_lambda, compute_xi, compute_beta, r0_from_density, B_M, KEV_TO_J
from src.quantities.pressure import compute_pressure, C_PRESSURE_QEOS
from src.quantities.internal_energy import compute_kinetic_energy

# ---------------------------------------------------------------------------
# Grid defaults: log-spaced, consistent with energy script
# ---------------------------------------------------------------------------
N_X   = 1024
X_MIN = 1e-6

# ---------------------------------------------------------------------------
# Training min and max, to know if any table values are being done with OOD inputs
# ---------------------------------------------------------------------------
T_1_MIN = 0.01 # kev
T_1_MAX = 10   # kev
ALPHA_1_MIN = 1
ALPHA_1_MAX = 10


# ---------------------------------------------------------------------------
# Core calculation for one (rho, T) point
# ---------------------------------------------------------------------------

def compute_dimensionless(
    rho: float,
    T_keV: float,
    Z: float,
    A: float,
    x_grid: np.ndarray,
    model,
    device: str,
) -> dict:
    """
    Compute dimensionless TF EoS ratios for a single (rho, T) pair.

    Returns dict with keys: alpha_1, T_1, Pv_kTZ, Ekin_kTZ, Epot_kTZ
    """
    # 1. density -> cell radius -> Z=1 reduced variables
    r0              = r0_from_density(rho, A)
    alpha_1, T_1    = z_scale_inputs(Z, r0, T_keV)

    # 2. FMT constants for FD argument
    gamma = compute_gamma(T_1)
    lam   = compute_lambda(alpha_1, T_1)

    # 3. PINN forward pass on full grid -> phi(x)
    phi   = predict_phi(alpha_1, T_1, x_grid, model, device)

    # 4. FD argument on full grid; boundary value
    beta = compute_beta(phi, gamma)
    beta_b = float(beta[-1])
    xi    = compute_xi(phi, x_grid, gamma, lam)
    xi_1  = float(xi[-1])

    # 5. Z=1 pressure (boundary only)
    p_result = compute_pressure(
        phi_boundary = float(phi[-1]),
        T_1_keV      = T_1,
        gamma        = gamma,
        lam          = lam,
        Z            = Z,
    )
    P_e1 = p_result["P_e1"]    # [Pa], Z=1 system

    # 6. Z=1 cell volume
    r_0_1 = alpha_1 * B_M
    V_1   = (4.0 / 3.0) * math.pi * r_0_1**3    # [m^3]

    # 7. Z=1 kinetic energy (full grid integral)
    K_1 = compute_kinetic_energy(xi, x_grid, T_1, alpha_1)    # [J]

    # 8. Common denominator: kT_1 in Joules (the Z=1 thermal energy)
    kT_1 = T_1 * KEV_TO_J    # [J]

    # 9. Dimensionless ratios
    Pv_kTZ   = (P_e1 * V_1) / kT_1          # Pv / kTZ  (Z cancels in Z=1 system)
    Ekin_kTZ = K_1 / kT_1                   # E_kin / kTZ
    Epot_kTZ = (3.0 * P_e1 * V_1 - 2.0 * K_1) / kT_1   # virial: U_pot / kTZ

    return {
        "alpha_1":  alpha_1,
        "T_1":      T_1,
        "Pv_kTZ":   Pv_kTZ,
        "Ekin_kTZ": Ekin_kTZ,
        "Epot_kTZ": Epot_kTZ,
        "P_e_mbar": p_result["P_e"] / 1e11,
        "beta_b":   beta_b
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="TF-EoS dimensionless quantity calculator (Pv/kTZ, Ekin/kTZ, Epot/kTZ)."
    )
    parser.add_argument("--Z",      type=float, required=True,
                        help="Atomic number (e.g. 26 for Fe)")
    parser.add_argument("--A",      type=float, required=True,
                        help="Atomic mass number (e.g. 56 for Fe56)")
    parser.add_argument("--rho",    type=float, nargs="+", required=True,
                        help="Mass density/densities [g/cm^3]")
    parser.add_argument("--T",      type=float, nargs="+", required=True,
                        help="Temperature(s) [keV]")
    parser.add_argument("--config", type=str,   default=None,
                        help="Path to EoS config yaml (default: configs/default.yaml)")
    parser.add_argument("--n_x",    type=int,   default=N_X,
                        help=f"Number of x grid points (default: {N_X})")
    parser.add_argument("--x_min",  type=float, default=X_MIN,
                        help=f"Minimum x grid value (default: {X_MIN})")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if len(args.rho) != len(args.T):
        raise ValueError(
            f"--rho and --T must have the same length, "
            f"got {len(args.rho)} and {len(args.T)}."
        )

    n_cases = len(args.rho)

    # Load model
    cfg    = load_eos_config(args.config)
    print("--- Config ---")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("--------------")
    device = cfg.get("device", "cpu")
    print("Loading PINN...")
    model, _ = load_pinn(cfg, verbose=True)

    # Build x grid once — same for all cases
    x_grid = build_x_grid(args.n_x, args.x_min)

    # Run all cases
    rows = []
    for i, (rho, T) in enumerate(zip(args.rho, args.T)):
        r0           = r0_from_density(rho, args.A)
        alpha_1, T_1 = z_scale_inputs(args.Z, r0, T)

        if not (T_1_MIN <= T_1 <= T_1_MAX):
            print(f"  [warn] case {i+1}: T_1 = {T_1:.4g} keV — likely OOD (training range ~{T_1_MIN}-{T_1_MAX} keV)")

        if not (ALPHA_1_MIN <= alpha_1 <= ALPHA_1_MAX):
            print(f"  [warn] case {i+1}: alpha_1 = {alpha_1:.4g} — likely OOD (training range ~{ALPHA_1_MIN}-{ALPHA_1_MAX})")


        result = compute_dimensionless(
            rho    = rho,
            T_keV  = T,
            Z      = args.Z,
            A      = args.A,
            x_grid = x_grid,
            model  = model,
            device = device,
        )
        rows.append((i + 1, rho, T, result["beta_b"], result["Pv_kTZ"], result["Ekin_kTZ"], result["Epot_kTZ"], result ["P_e_mbar"], result["T_1"], result["alpha_1"]))

    # Print table
    col_w = [6, 12, 12, 12, 12, 12, 12, 12, 12, 12]
    header = (
        f"{'Case':>{col_w[0]}}"
        f"{'rho (g/cc)':>{col_w[1]}}"
        f"{'T (keV)':>{col_w[2]}}"
        f"{'Beta_b':>{col_w[3]}}"
        f"{'Pv/kTZ':>{col_w[4]}}"
        f"{'Ekin/kTZ':>{col_w[5]}}"
        f"{'Epot/kTZ':>{col_w[6]}}"
        f"{"P (Mbar)":>{col_w[7]}}"
        f"{'T_1 (keV)':>{col_w[8]}}"
        f"{'alpha_1':>{col_w[9]}}"
    )
    separator = "-" * sum(col_w)

    print(f"\n{'TF EoS Dimensionless Quantities':^{sum(col_w)}}")
    print(f"Z = {int(args.Z)},  A = {int(args.A)}")
    print(separator)
    print(header)
    print(separator)

    for case, rho, T, beta_b, pv, ek, ep, pmbar, t1, a1 in rows:
        print(
            f"{case:>{col_w[0]}}"
            f"{rho:>{col_w[1]}.4f}"
            f"{T:>{col_w[2]}.4f}"
            f"{beta_b:>{col_w[3]}.4f}"
            f"{pv:>{col_w[4]}.4f}"
            f"{ek:>{col_w[5]}.4f}"
            f"{ep:>{col_w[6]}.4f}"
            f"{pmbar:>{col_w[7]}.4f}"
            f"{t1:>{col_w[8]}.4f}"
            f"{a1:>{col_w[9]}.4f}"
        )

    print(separator)


if __name__ == "__main__":
    main()