"""
Electron Helmholtz free energy for the Thomas-Fermi EoS.

Formula (Z=1 reduced system, thermodynamic identity F = E - T*S = mu - P*V):

    F_Z1 = mu_e1  -  2/3 * K_1  -  U_ee_1

All energy inputs (K_1, mu_e1, U_ee_1) must be in [J].

Z-scaling (FMT Section VI):
    F_physical = Z^(7/3) * F_Z1

Same exponent as internal energy E, since F = E - T*S and
T*S ~ Z^(4/3) * Z = Z^(7/3).

Unit conversion: 1 J/kg = 1e4 erg/g  (consistent with internal_energy.py).
"""

from ..inputs import M_PROTON


def compute_helmholtz(
    K_1: float,
    U_ee_1: float,
    mu_e1: float,
    Z: float,
    A: float,
) -> dict:
    """
    Electron Helmholtz free energy in Z=1 system and scaled to physical Z.

        F_Z1 = mu_e1  -  2/3 * K_1  -  U_ee_1

    Args:
        K_1:    Z=1 kinetic energy                [J]
        U_ee_1: Z=1 electron-electron energy      [J]
        mu_e1:  Z=1 electron chemical potential   [J]
        Z:      atomic number of physical element
        A:      atomic mass number

    Returns:
        dict with keys:
            'F_Z1'      : Z=1 Helmholtz free energy per atom  [J]
            'F_phys'    : physical Helmholtz free energy per atom [J]
            'F_J_kg'    : physical Helmholtz free energy per mass [J/kg]
            'F_erg_g'   : physical Helmholtz free energy per mass [erg/g]
    """
    F_Z1 = mu_e1  -  (2.0 / 3.0) * K_1  -  U_ee_1

    # Z-scaling: same exponent as internal energy (FMT Section VI)
    F_phys = Z**(7.0 / 3.0) * F_Z1                 # [J] per atom

    F_J_kg  = F_phys / (A * M_PROTON)               # [J/kg]
    F_erg_g = F_J_kg * 1e4                          # [erg/g]

    return {
        "F_Z1":    F_Z1,
        "F_phys":  F_phys,
        "F_J_kg":  F_J_kg,
        "F_erg_g": F_erg_g,
    }
