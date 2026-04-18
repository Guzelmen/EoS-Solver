"""
FMT Table XI comparison script.

Reproduces FMT paper Table XI for Fe (Z=26, A=56) and compares PINN predictions
against the published FMT reference values.

Columns: Case, rho, T, alpha_1, T_1, beta_b,
         Pv_fmt, Pv_pinn, Pv_err,
         Epot_fmt, Epot_pinn, Epot_err,
         Ekin_fmt, Ekin_pinn, Ekin_err

Usage:
    python -m scripts.complete_fmt_table_xi --config configs/default.yaml
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_loader import load_eos_config, load_pinn, build_x_grid, predict_phi
from src.inputs import z_scale_inputs, compute_gamma, compute_lambda, compute_xi, compute_beta, r0_from_density, B_M, KEV_TO_J
from src.quantities.pressure import compute_pressure
from src.quantities.internal_energy import compute_kinetic_energy

# ---------------------------------------------------------------------------
# Fixed element: Fe
# ---------------------------------------------------------------------------
Z_FE = 26.0
A_FE = 56.0

# ---------------------------------------------------------------------------
# FMT Table XI reference data for Fe (Z=26, A=56)
# (rho [g/cc], T [keV], Pv/kTZ, Ekin/kTZ, Epot/kTZ)
# ---------------------------------------------------------------------------
FMT_TABLE_XI = [
    # rho,     T,       Pv/kTZ,  Ekin/kTZ,  Epot/kTZ
    (11.874,  0.2231,  0.5227,   7.5715,   -13.5750),
    (11.455,  0.4926,  0.7028,   3.7802,    -5.4520),
    ( 0.8974, 0.1476,  0.5120,  10.8820,   -20.2006),
    ( 0.7644, 0.2381,  0.6470,   6.7993,   -11.6576),
    ( 0.9280, 0.5297,  0.8283,   3.3600,    -4.2351),
    (26.28,   0.2366,  0.5266,   7.2902,   -13.0006),
    (54.62,  14.660,   0.9936,   0.7742,    -0.0676),
    ( 4.154,  0.2923,  0.6239,   5.7792,    -9.6867),
    ( 1.030,  0.9892,  0.9172,   2.2511,    -1.7505),
    ( 0.2996, 0.3416,  0.8037,   4.7536,    -7.0961),
    ( 0.5235, 0.0326,  0.2380,  49.0072,   -97.3000),
]

# ---------------------------------------------------------------------------
# Grid defaults
# ---------------------------------------------------------------------------
N_X   = 256
X_MIN = 1e-6

# ---------------------------------------------------------------------------
# Training range bounds per stage (alpha_min, alpha_max, T_min, T_max)
# ---------------------------------------------------------------------------
STAGE_RANGES = {
    1: (1.0,  5.0,  1.0,   10.0),
    2: (1.0, 10.0,  0.1,   10.0),
    3: (1.0, 20.0,  0.01,  10.0),
    4: (1.0, 40.0,  0.001, 10.0),
}


def compute_dimensionless(rho, T_keV, x_grid, model, device):
    r0           = r0_from_density(rho, A_FE)
    alpha_1, T_1 = z_scale_inputs(Z_FE, r0, T_keV)

    gamma = compute_gamma(T_1)
    lam   = compute_lambda(alpha_1, T_1)

    phi    = predict_phi(alpha_1, T_1, x_grid, model, device)
    beta   = compute_beta(phi, gamma)
    beta_b = float(beta[-1])
    xi     = compute_xi(phi, x_grid, gamma, lam)

    p_result = compute_pressure(
        phi_boundary=float(phi[-1]),
        T_1_keV=T_1,
        gamma=gamma,
        lam=lam,
        Z=Z_FE,
    )
    P_e1 = p_result["P_e1"]

    r_0_1 = alpha_1 * B_M
    V_1   = (4.0 / 3.0) * math.pi * r_0_1**3

    K_1  = compute_kinetic_energy(xi, x_grid, T_1, alpha_1)
    kT_1 = T_1 * KEV_TO_J

    Pv_kTZ   = (P_e1 * V_1) / kT_1
    Ekin_kTZ = K_1 / kT_1
    Epot_kTZ = (3.0 * P_e1 * V_1 - 2.0 * K_1) / kT_1

    return {
        "alpha_1":  alpha_1,
        "T_1":      T_1,
        "beta_b":   beta_b,
        "Pv_kTZ":   Pv_kTZ,
        "Ekin_kTZ": Ekin_kTZ,
        "Epot_kTZ": Epot_kTZ,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="FMT Table XI comparison for Fe (Z=26, A=56)."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to EoS config yaml (default: configs/default.yaml)")
    parser.add_argument("--n_x",   type=int,   default=N_X,
                        help=f"Number of x grid points (default: {N_X})")
    parser.add_argument("--x_min", type=float, default=X_MIN,
                        help=f"Minimum x grid value (default: {X_MIN})")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Training stage — sets OOD bounds (default: 1)")
    parser.add_argument("--force_ranges", action="store_true",
                        help="Skip cases with OOD alpha_1 or T_1 instead of warning")
    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_eos_config(args.config)
    print("--- Config ---")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
    print("--------------")
    device = cfg.get("device", "cpu")
    print("Loading PINN...")
    model, _ = load_pinn(cfg, verbose=True)

    x_grid = build_x_grid(args.n_x, args.x_min)

    alpha_min, alpha_max, t1_min, t1_max = STAGE_RANGES[args.stage]
    print(f"Stage {args.stage} ranges: alpha_1 [{alpha_min}, {alpha_max}], T_1 [{t1_min}, {t1_max}] keV")

    rows = []
    for i, (rho, T, pv_fmt, ekin_fmt, epot_fmt) in enumerate(FMT_TABLE_XI):
        r0           = r0_from_density(rho, A_FE)
        alpha_1, T_1 = z_scale_inputs(Z_FE, r0, T)

        t1_ood    = not (t1_min <= T_1 <= t1_max)
        alpha_ood = not (alpha_min <= alpha_1 <= alpha_max)

        if args.force_ranges and (t1_ood or alpha_ood):
            print(f"  [skip] case {i+1}: OOD (alpha_1={alpha_1:.4g}, T_1={T_1:.4g}) — excluded by --force_ranges")
            continue

        if t1_ood:
            print(f"  [warn] case {i+1}: T_1 = {T_1:.4g} keV — likely OOD (stage {args.stage} range ~{t1_min}-{t1_max} keV)")
        if alpha_ood:
            print(f"  [warn] case {i+1}: alpha_1 = {alpha_1:.4g} — likely OOD (stage {args.stage} range ~{alpha_min}-{alpha_max})")

        res = compute_dimensionless(rho, T, x_grid, model, device)

        rows.append({
            "case":      i + 1,
            "rho":       rho,
            "T":         T,
            "alpha_1":   res["alpha_1"],
            "T_1":       res["T_1"],
            "beta_b":    res["beta_b"],
            "pv_fmt":    pv_fmt,
            "pv_pinn":   res["Pv_kTZ"],
            "pv_err":     abs(res["Pv_kTZ"] - pv_fmt),
            "pv_pct":     abs(res["Pv_kTZ"] - pv_fmt) / abs(pv_fmt) * 100,
            "epot_fmt":   epot_fmt,
            "epot_pinn":  res["Epot_kTZ"],
            "epot_err":   abs(res["Epot_kTZ"] - epot_fmt),
            "epot_pct":   abs(res["Epot_kTZ"] - epot_fmt) / abs(epot_fmt) * 100,
            "ekin_fmt":   ekin_fmt,
            "ekin_pinn":  res["Ekin_kTZ"],
            "ekin_err":   abs(res["Ekin_kTZ"] - ekin_fmt),
            "ekin_pct":   abs(res["Ekin_kTZ"] - ekin_fmt) / abs(ekin_fmt) * 100,
        })

    # ---------------------------------------------------------------------------
    # Print table — two header rows to keep column widths reasonable
    # ---------------------------------------------------------------------------
    # beta_b up to ~-1.7e7 (16 chars); T_1 printed to 6dp needs 10; alpha_1 needs 9
    # pct columns width 8 (e.g. "1234.567")
    cw = [5, 8, 7, 9, 10, 16,  9, 9, 8, 8,  10, 10, 9, 8,  10, 10, 9, 8]
    total = sum(cw)

    h1 = (
        f"{'Case':>{cw[0]}}"
        f"{'rho':>{cw[1]}}"
        f"{'T':>{cw[2]}}"
        f"{'alpha_1':>{cw[3]}}"
        f"{'T_1':>{cw[4]}}"
        f"{'beta_b':>{cw[5]}}"
        f"{'Pv_fmt':>{cw[6]}}"
        f"{'Pv_pinn':>{cw[7]}}"
        f"{'Pv_err':>{cw[8]}}"
        f"{'Pv_%':>{cw[9]}}"
        f"{'Epot_fmt':>{cw[10]}}"
        f"{'Epot_pinn':>{cw[11]}}"
        f"{'Epot_err':>{cw[12]}}"
        f"{'Epot_%':>{cw[13]}}"
        f"{'Ekin_fmt':>{cw[14]}}"
        f"{'Ekin_pinn':>{cw[15]}}"
        f"{'Ekin_err':>{cw[16]}}"
        f"{'Ekin_%':>{cw[17]}}"
    )
    sep = "-" * total

    print(f"\n{'FMT Table XI Comparison  —  Fe (Z=26, A=56)':^{total}}")
    print(f"{'All dimensionless quantities in units of kTZ':^{total}}")
    print(sep)
    print(h1)
    print(sep)

    for r in rows:
        print(
            f"{r['case']:>{cw[0]}}"
            f"{r['rho']:>{cw[1]}.3f}"
            f"{r['T']:>{cw[2]}.3f}"
            f"{r['alpha_1']:>{cw[3]}.3f}"
            f"{r['T_1']:>{cw[4]}.6f}"
            f"{r['beta_b']:>{cw[5]}.3f}"
            f"{r['pv_fmt']:>{cw[6]}.3f}"
            f"{r['pv_pinn']:>{cw[7]}.3f}"
            f"{r['pv_err']:>{cw[8]}.3f}"
            f"{r['pv_pct']:>{cw[9]}.2f}"
            f"{r['epot_fmt']:>{cw[10]}.3f}"
            f"{r['epot_pinn']:>{cw[11]}.3f}"
            f"{r['epot_err']:>{cw[12]}.3f}"
            f"{r['epot_pct']:>{cw[13]}.2f}"
            f"{r['ekin_fmt']:>{cw[14]}.3f}"
            f"{r['ekin_pinn']:>{cw[15]}.3f}"
            f"{r['ekin_err']:>{cw[16]}.3f}"
            f"{r['ekin_pct']:>{cw[17]}.2f}"
        )

    print(sep)


if __name__ == "__main__":
    main()
