"""
Electron energies for the Thomas-Fermi EoS.

Three contributions per atom (QEOS Eq. 78, More et al. 1988):

    E_e = (K + U_en + U_ee) / (A * m_p)    [J/kg -> erg/g]

All quantities computed in Z=1 reduced system, then Z-scaled.
Z-scaling: total energy per atom scales as Z^(7/3) (FMT Section VI).

Kinetic energy (QEOS Eq. 75):
    K_1 = C_K * r_0_1^3 * T_1_keV^(5/2) * integral_0^1 F_{3/2}(xi(x)) x^2 dx
    C_K = C1 * 4pi * KEV_TO_J^(5/2)                [m^-3 J]

Electron-nucleus energy (QEOS Eq. 76, Z=1):
    U_en_1 = -C_UEN * r_0_1^2 * T_1_keV^(3/2) * integral_0^1 F_{1/2}(xi(x)) x dx
    C_UEN = C1 * 4pi * KC * KEV_TO_J^(3/2)          [m^-2 J]

Electron-electron energy via virial theorem (FMT Section VI):
    2K + U_en + U_ee = 3 * P_1 * V_1
    => U_ee_1 = 3 * P_1 * V_1 - 2*K_1 - U_en_1
    where P_1 = (2/3)*C1*KEV_TO_J^(5/2) * T_1_keV^(5/2) * F_{3/2}(xi_1)  (QEOS Eq. 81)
          V_1 = (4pi/3) * r_0_1^3

Units check:
    C_K   : C1[m^-3 J^-3/2] * KEV_TO_J^(5/2)[J^(5/2)]        = [m^-3 J]
    C_UEN : C1[m^-3 J^-3/2] * KC[J m] * KEV_TO_J^(3/2)[J^(3/2)] = [m^-2 J]
    K_1   : [m^-3 J] * r_0_1^3[m^3] * T^(5/2)[1]             = [J]  ✓
    U_en_1: [m^-2 J] * r_0_1^2[m^2] * T^(3/2)[1]             = [J]  ✓
"""

import math
import numpy as np
import torch

from ..inputs import C1, KC, KEV_TO_J, B_M, M_PROTON
from ..fd_integrals import fermi_dirac_half, fermi_dirac_three_half

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
C_K        = C1 * 4.0 * math.pi * KEV_TO_J**2.5     # kinetic prefactor    [m^-3 J]
C_UEN      = C1 * 4.0 * math.pi * KC * KEV_TO_J**1.5 # en-nucleus prefactor [m^-2 J]
C_PRESSURE = (2.0 / 3.0) * C1 * KEV_TO_J**2.5        # pressure prefactor   [Pa keV^-5/2]
                                                       # consistent with QEOS Eq. 81


def compute_kinetic_energy(
    xi: np.ndarray,
    x: np.ndarray,
    T_1_keV: float,
    alpha_1: float,
) -> float:
    """
    Electron kinetic energy in Z=1 system (QEOS Eq. 75).

        K_1 = C_K * r_0_1^3 * T_1_keV^(5/2) * integral_0^1 F_{3/2}(xi(x)) x^2 dx

    Args:
        xi:      FD argument gamma*phi/(lambda*x) at each grid point [n_x]
        x:       grid points in (0, 1]                               [n_x]
        T_1_keV: Z=1 reduced temperature                             [keV]
        alpha_1: Z=1 reduced cell radius                             [dimensionless]

    Returns:
        K_1: kinetic energy in Z=1 system [J]
    """
    r_0_1 = alpha_1 * B_M

    xi_t      = torch.tensor(xi, dtype=torch.float32)
    F32       = fermi_dirac_three_half(xi_t).numpy()
    integral  = np.trapz(F32 * x**2, x)

    return C_K * r_0_1**3 * T_1_keV**2.5 * integral


def compute_en_energy(
    xi: np.ndarray,
    x: np.ndarray,
    T_1_keV: float,
    alpha_1: float,
) -> float:
    """
    Electron-nucleus Coulomb energy in Z=1 system (QEOS Eq. 76, Z=1).

        U_en_1 = -C_UEN * r_0_1^2 * T_1_keV^(3/2) * integral_0^1 F_{1/2}(xi(x)) x dx

    The 1/r Coulomb factor reduces the r_0_1 power from 3 to 2 relative to K_1,
    and the missing kT factor reduces the KEV_TO_J power from 5/2 to 3/2.

    Args:
        xi:      FD argument at each grid point [n_x]
        x:       grid points in (0, 1]          [n_x]
        T_1_keV: Z=1 reduced temperature        [keV]
        alpha_1: Z=1 reduced cell radius        [dimensionless]

    Returns:
        U_en_1: electron-nucleus energy in Z=1 system [J] (negative)
    """
    r_0_1 = alpha_1 * B_M

    xi_t      = torch.tensor(xi, dtype=torch.float32)
    F12       = fermi_dirac_half(xi_t).numpy()
    integral  = np.trapz(F12 * x, x)

    return -C_UEN * r_0_1**2 * T_1_keV**1.5 * integral


def compute_ee_energy_virial(
    K_1: float,
    U_en_1: float,
    xi_1: float,
    T_1_keV: float,
    alpha_1: float,
) -> float:
    """
    Electron-electron energy via virial theorem (FMT Section VI).

    The virial theorem in the TF model (FMT Section VI, Eq. 29):
        2*K + U_en + U_ee = 3 * P_1 * V_1
    Rearranging:
        U_ee_1 = 3 * P_1 * V_1 - 2*K_1 - U_en_1

    P_1 is the Z=1 electron pressure (QEOS Eq. 81):
        P_1 = C_PRESSURE * T_1_keV^(5/2) * F_{3/2}(xi_1)

    Args:
        K_1:     kinetic energy from compute_kinetic_energy        [J]
        U_en_1:  electron-nucleus energy from compute_en_energy    [J]
        xi_1:    FD argument at boundary = mu_e1 / kT_1
        T_1_keV: Z=1 reduced temperature                          [keV]
        alpha_1: Z=1 reduced cell radius                          [dimensionless]

    Returns:
        U_ee_1: electron-electron energy in Z=1 system [J]
    """
    r_0_1 = alpha_1 * B_M
    V_1   = (4.0 / 3.0) * math.pi * r_0_1**3

    xi_t  = torch.tensor([[xi_1]], dtype=torch.float32)
    F32_b = float(fermi_dirac_three_half(xi_t).item())
    P_1   = C_PRESSURE * T_1_keV**2.5 * F32_b

    return 3.0 * P_1 * V_1 - 2.0 * K_1 - U_en_1


def compute_total_energy(
    xi: np.ndarray,
    x: np.ndarray,
    xi_1: float,
    T_1_keV: float,
    alpha_1: float,
    Z: float,
    A: float,
) -> dict:
    """
    Total electron internal energy per gram (QEOS Eq. 78).

        E_e = Z^(7/3) * (K_1 + U_en_1 + U_ee_1) / (A * m_p)

    Z-scaling: FMT Section VI shows total energy per atom scales as Z^(7/3).

    Unit conversion: 1 J/kg = 1e4 erg/g  (since 1 J = 1e7 erg, 1 kg = 1e3 g)

    Args:
        xi:      FD argument on full grid  [n_x]
        x:       grid points               [n_x]
        xi_1:    FD argument at boundary   [dimensionless]
        T_1_keV: Z=1 reduced temperature   [keV]
        alpha_1: Z=1 reduced cell radius   [dimensionless]
        Z:       atomic number
        A:       atomic mass number

    Returns:
        dict with keys:
            'K_1'       : Z=1 kinetic energy              [J]
            'U_en_1'    : Z=1 electron-nucleus energy     [J]
            'U_ee_1'    : Z=1 electron-electron energy    [J]
            'E_e_J_kg'  : total electron energy per mass  [J/kg]
            'E_e_erg_g' : total electron energy per mass  [erg/g]
    """
    K_1    = compute_kinetic_energy(xi, x, T_1_keV, alpha_1)
    U_en_1 = compute_en_energy(xi, x, T_1_keV, alpha_1)
    U_ee_1 = compute_ee_energy_virial(K_1, U_en_1, xi_1, T_1_keV, alpha_1)

    E_atom_1  = K_1 + U_en_1 + U_ee_1
    E_atom    = Z**(7.0 / 3.0) * E_atom_1      # physical Z, per atom [J]

    E_e_J_kg  = E_atom / (A * M_PROTON)         # [J/kg]
    E_e_erg_g = E_e_J_kg * 1e4                  # [erg/g]

    return {
        "K_1":       K_1,
        "U_en_1":    U_en_1,
        "U_ee_1":    U_ee_1,
        "E_e_J_kg":  E_e_J_kg,
        "E_e_erg_g": E_e_erg_g,
    }