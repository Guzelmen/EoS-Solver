"""
Electron entropy for the Thomas-Fermi EoS.

Formula (Z=1 reduced system, derived from thermodynamic identity S = (E - F) / T):

    S_Z1 = (5/3 * K_1  -  mu_e1  +  U_en_1  +  2 * U_ee_1) / (kT_1)

where kT_1 = T_1_keV * KEV_TO_J  [J].

All energy inputs (K_1, mu_e1, U_en_1, U_ee_1) must be in [J].
The result S_Z1 is dimensionless (= S / k_B per atom in the Z=1 system).
To obtain physical entropy [J/K] per atom, multiply by k_B.

Z-scaling (FMT Section VI):
    S_physical = Z * S_Z1

The Z^1 factor follows from S ~ E / T ~ Z^(7/3) / Z^(4/3) = Z.
"""

import math

from ..inputs import KEV_TO_J, M_PROTON

K_B = 1.380649e-23          # Boltzmann constant [J/K]


def compute_entropy(
    K_1: float,
    U_en_1: float,
    U_ee_1: float,
    mu_e1: float,
    T_1_keV: float,
    Z: float,
    A: float,
) -> dict:
    """
    Electron entropy in Z=1 system and scaled to physical Z.

        S_Z1 = (5/3 * K_1  -  mu_e1  +  U_en_1  +  2 * U_ee_1) / (kT_1)

    Args:
        K_1:     Z=1 kinetic energy                [J]
        U_en_1:  Z=1 electron-nucleus energy       [J]
        U_ee_1:  Z=1 electron-electron energy      [J]
        mu_e1:   Z=1 electron chemical potential   [J]
        T_1_keV: Z=1 reduced temperature           [keV]
        Z:       atomic number of physical element
        A:       atomic mass number

    Returns:
        dict with keys:
            'S_Z1'      : Z=1 entropy per atom, dimensionless (S/k_B)  [1]
            'S_phys'    : physical entropy per atom, dimensionless      [1]
            'S_J_kg_K'  : physical entropy per unit mass                [J/(kg K)]
            'S_erg_g_K' : physical entropy per unit mass                [erg/(g K)]
    """
    kT_1 = T_1_keV * KEV_TO_J                      # [J]

    S_Z1 = (5.0 / 3.0 * K_1  -  mu_e1  +  U_en_1  +  2.0 * U_ee_1) / kT_1

    # Z-scaling: S_physical = Z * S_Z1
    S_phys = Z * S_Z1

    # Per unit mass: multiply by k_B to get [J/K per atom], then divide by (A * m_p)
    S_J_kg_K  = S_phys * K_B / (A * M_PROTON)      # [J/(kg K)]
    S_erg_g_K = S_J_kg_K * 1e4                      # [erg/(g K)]

    return {
        "S_Z1":       S_Z1,
        "S_phys":     S_phys,
        "S_J_kg_K":   S_J_kg_K,
        "S_erg_g_K":  S_erg_g_K,
    }
