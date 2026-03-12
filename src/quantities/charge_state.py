"""
Electron charge state (mean ionization) for the Thomas-Fermi EoS.

Source: QEOS Eq. 82 (More et al. 1988):

    Q = (4pi/3) * R0^3 * n(R0)

Physical meaning: number of free electrons per ion, estimated as the electron
number density at the cell boundary times the full cell volume. Q in [0, Z].

Derivation in our variables:
    n(R0) = C1 * (kT_1)^(3/2) * F_{1/2}(xi_1)     [m^-3]  (QEOS Eq. 70 at x=1)
    V_1   = (4pi/3) * r_0_1^3                       [m^3]

    Q_1 = V_1 * n(R0)
        = (4pi/3) * r_0_1^3 * C1 * (T_1_keV * KEV_TO_J)^(3/2) * F_{1/2}(xi_1)

Z-scaling (FMT Section VI):
    Q = Z * Q_1

Boundary-only quantity — only xi_1 = gamma * phi(1) / lambda is needed,
same as pressure and chemical potential. No grid integral required.
"""

import math
import torch

from ..inputs import C1, KEV_TO_J, B_M
from ..fd_integrals import fermi_dirac_half

# ---------------------------------------------------------------------------
# Module-level constant
# ---------------------------------------------------------------------------
# C_Q = C1 * (4pi/3) * KEV_TO_J^(3/2)   [m^-3 J^(-3/2) * m^3 * J^(3/2)] = [dimensionless per keV^(3/2)]
# Q_1 = C_Q * r_0_1^3 * T_1_keV^(3/2) * F_{1/2}(xi_1)
C_Q = C1 * (4.0 / 3.0) * math.pi * KEV_TO_J**1.5


def compute_charge_state(
    phi_boundary: float,
    T_1_keV: float,
    gamma: float,
    lam: float,
    alpha_1: float,
    Z: float,
) -> dict:
    """
    Compute electron charge state Q (mean ionization) per ion.

    Args:
        phi_boundary: phi(x=1) from PINN              [dimensionless]
        T_1_keV:      Z=1 reduced temperature          [keV]
        gamma:        0.0899 / T_1_keV^(3/4)          [dimensionless]
        lam:          alpha_1 * b * T_1_keV^(1/4) / C0 [dimensionless]
        alpha_1:      Z=1 reduced cell radius          [dimensionless]
        Z:            atomic number of physical element

    Returns:
        dict with keys:
            'xi_1' : FD argument at boundary = mu_e1 / kT_1  [dimensionless]
            'F12'  : F_{1/2}(xi_1)                           [dimensionless]
            'Q_1'  : Z=1 charge state                        [dimensionless, in (0,1)]
            'Q'    : physical charge state                    [dimensionless, in (0,Z)]
    """
    r_0_1 = alpha_1 * B_M

    # FD argument at boundary (same as pressure and chemical potential)
    xi_1 = gamma * phi_boundary / lam

    # F_{1/2}(xi_1)
    xi_t = torch.tensor([[xi_1]], dtype=torch.float32)
    F12  = float(fermi_dirac_half(xi_t).item())

    # Z=1 charge state (QEOS Eq. 82 in Z=1 system)
    Q_1 = C_Q * r_0_1**3 * T_1_keV**1.5 * F12

    # Scale to physical Z (FMT Section VI)
    Q = Z * Q_1

    return {
        "xi_1": xi_1,
        "F12":  F12,
        "Q_1":  Q_1,
        "Q":    Q,
    }