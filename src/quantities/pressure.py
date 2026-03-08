"""
Electron pressure for the Thomas-Fermi EoS.

Formula (derived from FMT Eq. 26, divided by v, Z=1, with eps_0 = 1e3 * e absorbed):

    P_e1 = C_PRESSURE * T_1_keV^(5/2) * F_{3/2}(xi_1)   [Pa, Z=1 system]
    P_e  = Z^(10/3) * P_e1                                [Pa, physical]

where:
    C_PRESSURE = 1e6 / (6 * pi * C0_M^2)     [Pa keV^{-5/2}]
    xi_1       = gamma * phi(1) / lambda      [dimensionless, FD argument at boundary]

Derivation of C_PRESSURE:
    Start from FMT Eq. 26 / v, Z=1:
        P = (kT * T_keV^(3/2)) / (6*pi * 0.0899 * C0_M^3) * F_{3/2}(xi_1)
    Substitute 0.0899 = e / (1e3 * C0_M) and kT = T_keV * 1e3 * e:
        (kT)^(5/2) = (T_keV * 1e3 * e)^(5/2) = T_keV^(5/2) * 1e^(15/2) * e^(5/2)
    The e^(5/2) and 1e^(3/2) from the denominator cancel, leaving:
        P = [1e6 / (6*pi * C0_M^2)] * T_keV^(5/2) * F_{3/2}(xi_1)

FD argument:
    xi_1 = gamma * phi(1) / lambda
    This equals beta_b / b in FMT notation (beta = gamma*phi, b_FMT = lambda),
    and equals mu_e / kT_1 by the gauge choice V(r0) = 0.

Z-scaling:
    P_e = Z^(10/3) * P_e1  from the FMT similarity transformation.
    All computation done in Z=1 reduced system, scaled once at the end.
"""

import math
import torch

from ..fd_integrals import fermi_dirac_three_half, C0_M
from ..inputs import KC, KEV_TO_J

# ---------------------------------------------------------------------------
# Pressure prefactor
# ---------------------------------------------------------------------------

# C_PRESSURE = 1e6 / (6 * pi * C0_M^2)
# Units: Pa keV^{-5/2}
# P [Pa] = C_PRESSURE * T_1_keV^(5/2) * F_{3/2}(xi_1)
C_PRESSURE_OLD = 1e6 / (6.0 * math.pi * C0_M**2)

C_PRESSURE = KEV_TO_J**2 / (6.0 * math.pi * KC * C0_M**2)


# ---------------------------------------------------------------------------
# Pressure calculation
# ---------------------------------------------------------------------------

def compute_pressure(
    phi_boundary: float,
    T_1_keV: float,
    gamma: float,
    lam: float,
    Z: float,
) -> dict:
    """
    Compute electron pressure in the Z=1 system and scale to physical Z.

    Args:
        phi_boundary: phi(x=1) from PINN [dimensionless]
        T_1_keV:      Z=1 reduced temperature [keV]
        gamma:        0.0899 / T_1_keV^(3/4) [dimensionless]
        lam:          alpha_1 * b * T_1_keV^(1/4) / C0 [dimensionless]
        Z:            atomic number of physical element

    Returns:
        dict with keys:
            'xi_1' : FD argument at boundary = mu_e / kT_1  [dimensionless]
            'F32'  : F_{3/2}(xi_1)                          [dimensionless]
            'P_e1' : Z=1 electron pressure                   [Pa]
            'P_e'  : physical electron pressure              [Pa]
    """
    # FD argument at boundary
    xi_1 = gamma * phi_boundary / lam

    # F_{3/2}(xi_1): feed as [1,1] tensor, return scalar
    xi_tensor = torch.tensor([[xi_1]], dtype=torch.float32)
    
    F32 = float(fermi_dirac_three_half(xi_tensor).item())

    # Z=1 pressure [Pa]
    P_e1 = C_PRESSURE * T_1_keV**2.5 * F32

    # Scale to physical Z [Pa]
    P_e = Z**(10.0 / 3.0) * P_e1

    return {
        "xi_1": xi_1,
        "F32":  F32,
        "P_e1": P_e1,
        "P_e":  P_e,
    }


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------

def pa_to_mbar(P_pa: float) -> float:
    """Convert pressure from Pa to Mbar.  1 Mbar = 1e11 Pa."""
    return P_pa / 1e11