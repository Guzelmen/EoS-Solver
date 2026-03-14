"""
TF-EoS pressure calculator from precomputed beta_b.

Given precomputed boundary values beta_b (from a solver or reference targets),
computes pressure without needing the full phi(x) profile or a PINN.

beta_b = gamma * phi(1)  =>  phi_boundary = beta_b / gamma
xi_1   = gamma * phi(1) / lambda = beta_b / lambda

Inputs are alpha_1 (Z=1 reduced cell radius) and T_1 (Z=1 reduced temperature),
the same variables used as PINN inputs.

Usage examples:
    # Single point
    python -m scripts.pressure_from_beta_b \\
        --Z 26 --A 56 \\
        --alpha_1 3.5 \\
        --T_1 0.564 \\
        --beta_b 1.234

    # Multiple points (all three lists must be the same length)
    python -m scripts.pressure_from_beta_b \\
        --Z 26 --A 56 \\
        --alpha_1 2.1 3.5 4.0 \\
        --T_1 0.1 0.5 1.0 \\
        --beta_b 0.1 0.2 0.3
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inputs import compute_gamma, compute_lambda, B_M, KEV_TO_J, density_from_alpha
from src.quantities.pressure import compute_pressure, C_PRESSURE_QEOS
from src.fd_integrals import fermi_dirac_three_half

# ---------------------------------------------------------------------------
# Training range bounds (for OOD warnings)
# ---------------------------------------------------------------------------
T_1_MIN     = 0.01
T_1_MAX     = 10
ALPHA_1_MIN = 1
ALPHA_1_MAX = 10


# ---------------------------------------------------------------------------
# Core calculation for one (rho, T, beta_b) point
# ---------------------------------------------------------------------------

def compute_pressure_from_beta_b(
    alpha_1: float,
    T_1: float,
    beta_b: float,
    Z: float,
    A: float,
) -> dict:
    """
    Compute TF pressure from a precomputed boundary value beta_b.

    beta_b = gamma * phi(1)  =>  phi_boundary = beta_b / gamma
    xi_1   = beta_b / lambda  (FD argument at cell boundary)

    Args:
        alpha_1: Z=1 reduced dimensionless cell radius
        T_1:     Z=1 reduced temperature [keV]
        beta_b:  precomputed gamma * phi(1)
        Z:       atomic number
        A:       atomic mass number (used only to recover rho for display)

    Returns dict with keys: rho, xi_1, Pv_kTZ, P_e1, P_e, P_e_mbar
    """
    gamma = compute_gamma(T_1)
    lam   = compute_lambda(alpha_1, T_1)

    # Recover phi_boundary from beta_b, then call standard pressure routine
    phi_boundary = beta_b / gamma

    p_result = compute_pressure(
        phi_boundary = phi_boundary,
        T_1_keV      = T_1,
        gamma        = gamma,
        lam          = lam,
        Z            = Z,
    )

    P_e1 = p_result["P_e1"]
    xi_1 = p_result["xi_1"]

    # Cell volume in Z=1 system
    r_0_1 = alpha_1 * B_M
    V_1   = (4.0 / 3.0) * math.pi * r_0_1**3    # [m^3]

    kT_1   = T_1 * KEV_TO_J
    Pv_kTZ = (P_e1 * V_1) / kT_1

    # Recover rho for display: invert alpha_1 = Z^(1/3) * alpha
    alpha = alpha_1 / Z**(1.0 / 3.0)
    rho   = density_from_alpha(alpha, A)

    return {
        "rho":      rho,
        "xi_1":     xi_1,
        "Pv_kTZ":   Pv_kTZ,
        "P_e1":     P_e1,
        "P_e":      p_result["P_e"],
        "P_e_mbar": p_result["P_e"] / 1e11,
    }


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="TF-EoS pressure calculator from precomputed beta_b values."
    )
    parser.add_argument("--Z",      type=float, required=True,
                        help="Atomic number (e.g. 26 for Fe)")
    parser.add_argument("--A",      type=float, required=True,
                        help="Atomic mass number (e.g. 56 for Fe56)")
    parser.add_argument("--alpha_1", type=float, nargs="+", required=True,
                        help="Z=1 reduced dimensionless cell radius (PINN input)")
    parser.add_argument("--T_1",    type=float, nargs="+", required=True,
                        help="Z=1 reduced temperature [keV] (PINN input)")
    parser.add_argument("--beta_b", type=float, nargs="+", required=True,
                        help="Precomputed boundary value beta_b = gamma * phi(1) [dimensionless]")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not (len(args.alpha_1) == len(args.T_1) == len(args.beta_b)):
        raise ValueError(
            f"--alpha_1, --T_1, and --beta_b must have the same length, "
            f"got {len(args.alpha_1)}, {len(args.T_1)}, and {len(args.beta_b)}."
        )

    rows = []
    for i, (alpha_1, T_1, beta_b) in enumerate(zip(args.alpha_1, args.T_1, args.beta_b)):
        if not (T_1_MIN <= T_1 <= T_1_MAX):
            print(f"  [warn] case {i+1}: T_1 = {T_1:.4g} keV — likely OOD (training range ~{T_1_MIN}-{T_1_MAX} keV)")

        if not (ALPHA_1_MIN <= alpha_1 <= ALPHA_1_MAX):
            print(f"  [warn] case {i+1}: alpha_1 = {alpha_1:.4g} — likely OOD (training range ~{ALPHA_1_MIN}-{ALPHA_1_MAX})")

        result = compute_pressure_from_beta_b(
            alpha_1 = alpha_1,
            T_1     = T_1,
            beta_b  = beta_b,
            Z       = args.Z,
            A       = args.A,
        )
        rows.append((
            i + 1, result["rho"], T_1, beta_b,
            result["Pv_kTZ"], result["P_e_mbar"],
            T_1, alpha_1, result["xi_1"],
        ))

    # Print table
    col_w  = [6, 12, 12, 12, 12, 12, 12, 12, 12]
    header = (
        f"{'Case':>{col_w[0]}}"
        f"{'rho (g/cc)':>{col_w[1]}}"
        f"{'T (keV)':>{col_w[2]}}"
        f"{'beta_b':>{col_w[3]}}"
        f"{'Pv/kTZ':>{col_w[4]}}"
        f"{'P (Mbar)':>{col_w[5]}}"
        f"{'T_1 (keV)':>{col_w[6]}}"
        f"{'alpha_1':>{col_w[7]}}"
        f"{'xi_1':>{col_w[8]}}"
    )
    separator = "-" * sum(col_w)

    print(f"\n{'TF EoS Pressure from Precomputed beta_b':^{sum(col_w)}}")
    print(f"Z = {int(args.Z)},  A = {int(args.A)}")
    print(separator)
    print(header)
    print(separator)

    for case, rho, T, beta_b, pv, pmbar, t1, a1, xi1 in rows:
        print(
            f"{case:>{col_w[0]}}"
            f"{rho:>{col_w[1]}.4f}"
            f"{T:>{col_w[2]}.4f}"
            f"{beta_b:>{col_w[3]}.4f}"
            f"{pv:>{col_w[4]}.4f}"
            f"{pmbar:>{col_w[5]}.4f}"
            f"{t1:>{col_w[6]}.4f}"
            f"{a1:>{col_w[7]}.4f}"
            f"{xi1:>{col_w[8]}.4f}"
        )

    print(separator)


if __name__ == "__main__":
    main()
