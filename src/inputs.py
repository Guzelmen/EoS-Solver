"""
Input processing for the TF-EoS solver.

Handles:
  - Physical constants
  - (Z, r0, T) -> reduced Z=1 variables (alpha_1, T_1)
  - Derived constants gamma and lambda for the FD integrals
  - x grid construction
  - xi(x) = gamma * phi(x) / (lambda * x)  [FD argument at each grid point]

Convention notes:
  - r0 is in metres
  - T is in keV throughout (converted to Joules only when computing physical energies)
  - alpha = r0 / b  (dimensionless cell radius in TF length units)
  - alpha_1 = Z^(1/3) * alpha  (Z=1 reduced system)
  - T_1 = T_keV / Z^(4/3)     (Z=1 reduced temperature, still in keV)
  - gamma = 0.0899 / T_1^(3/4)           (dimensionless, boundary value of beta at origin)
  - lam   = alpha_1 * b * T_1^(1/4) / C0 (dimensionless length scale ratio)
  - xi(x) = gamma * phi(x) / (lam * x)   (argument of FD integrals, always > 0)
  - At boundary x=1: xi(1) = gamma * phi(1) / lam = mu_e / (k_B * T_1)
"""

import math
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (SI)
# ---------------------------------------------------------------------------
A0_M       = 5.291772105e-11          # Bohr radius [m]
B_M        = 0.25 * (4.5 * math.pi**2)**(1/3) * A0_M   # TF length scale b [m]
C0_M       = 1.602e-11                # Temperature length scale coefficient [m]
KB_J       = 1.380649e-23             # Boltzmann constant [J/K]
KB_KEV     = 8.617333e-8              # Boltzmann constant [keV/K]
KEV_TO_J   = 1.602176634e-16          # 1 keV in Joules
M_E        = 9.1093837015e-31         # Electron mass [kg]
HBAR       = 1.054571817e-34          # Reduced Planck constant [J·s]
E_CHARGE   = 1.602176634e-19          # Elementary charge [C]
EPS0       = 8.8541878128e-12         # Vacuum permittivity [F/m]
KC         = E_CHARGE**2 / (4 * math.pi * EPS0)  # Coulomb constant ke^2 [J·m]
M_PROTON   = 1.67262192369e-27        # proton mass [kg]

# Density of states prefactor c1 = (1/2pi^2) * (2*m_e/hbar^2)^(3/2)  [m^-3 J^-3/2]
C1         = (1.0 / (2.0 * math.pi**2)) * (2.0 * M_E / HBAR**2)**1.5

# ---------------------------------------------------------------------------
# Z-scaling and reduced variable transforms
# ---------------------------------------------------------------------------

def compute_alpha(r0: float) -> float:
    """
    Compute dimensionless cell radius alpha = r0 / b.

    Args:
        r0: cell radius [m]

    Returns:
        alpha: dimensionless  (no Z dependence)
    """
    return r0 / B_M


def z_scale_inputs(Z: float, r0: float, T_keV: float):
    """
    Map physical inputs (Z, r0, T) to the Z=1 reduced system.

    The TF similarity transformation gives:
        alpha_1 = Z^(1/3) * (r0/b)
        T_1     = T_keV / Z^(4/3)

    Args:
        Z:      atomic number (int or float)
        r0:     cell radius [m]
        T_keV:  temperature [keV]

    Returns:
        alpha_1: Z=1 reduced dimensionless cell radius
        T_1:     Z=1 reduced temperature [keV]
    """
    alpha = compute_alpha(r0)
    alpha_1 = Z**(1/3) * alpha
    T_1 = T_keV / Z**(4/3)
    return alpha_1, T_1


def compute_gamma(T_1_keV: float, Z_model: float = 1.0) -> float:
    """
    Compute gamma = 0.0899 * Z_model / T_1^(3/4).

    This is the boundary value beta(0) = gamma, encoding the nuclear
    singularity strength. For our Z=1 model, Z_model=1 always.

    Args:
        T_1_keV: temperature in the Z=1 system [keV]
        Z_model: atomic number of the model system (always 1 here)

    Returns:
        gamma: dimensionless
    """
    return 0.0899 * Z_model / T_1_keV**0.75


def compute_lambda(alpha_1: float, T_1_keV: float) -> float:
    """
    Compute lambda = alpha_1 * b * T_1^(1/4) / C0.

    Lambda is the dimensionless cell boundary in the FMT s-coordinate.
    The physical cell boundary is at s = lambda (i.e. x=1 maps to s=lambda).

    Args:
        alpha_1: Z=1 reduced dimensionless cell radius
        T_1_keV: Z=1 reduced temperature [keV]

    Returns:
        lam: dimensionless
    """
    return alpha_1 * B_M * T_1_keV**0.25 / C0_M


# ---------------------------------------------------------------------------
# x grid
# ---------------------------------------------------------------------------

def build_x_grid(n_x: int, x_min: float) -> np.ndarray:
    """
    Build a log-spaced grid from x_min to 1.

    Log spacing clusters points near x=0 where phi(x) and xi(x) vary most
    rapidly (xi -> infinity as x -> 0 near the nucleus).

    Args:
        n_x:   number of grid points
        x_min: smallest x value (avoid true zero due to 1/x in xi)

    Returns:
        x: 1D numpy array of shape [n_x], values in [x_min, 1]
    """
    return np.logspace(np.log10(x_min), 0.0, n_x)


# ---------------------------------------------------------------------------
# FD argument xi(x)
# ---------------------------------------------------------------------------

def compute_xi(phi: np.ndarray, x: np.ndarray,
               gamma: float, lam: float) -> np.ndarray:
    """
    Compute xi(x) = gamma * phi(x) / (lam * x).

    This is (mu_e - V_elec(r)) / kT at each grid point — the argument
    fed into all Fermi-Dirac integrals. Always positive inside the cell.

    At the boundary x=1:  xi(1) = gamma * phi(1) / lam = mu_e / kT_1

    Args:
        phi:   model output phi(x), shape [n_x]
        x:     grid points, shape [n_x]
        gamma: dimensionless constant (from compute_gamma)
        lam:   dimensionless constant (from compute_lambda)

    Returns:
        xi: shape [n_x], always > 0
    """
    return gamma * phi / (lam * x)


# for density stuff

# ---------------------------------------------------------------------------
# Density <-> r0 conversions
# ---------------------------------------------------------------------------
# Note: A (atomic mass number) appears ONLY here, not in the pressure formula.
# Pressure is a purely electronic quantity; A sets the ion mass that converts
# cell volume to mass density for the x-axis of P vs rho plots.


def r0_from_alpha(alpha_1: float) -> float:
    """Cell radius from dimensionless alpha_1. r0_1 = alpha_1 * b  [m]"""
    return alpha_1 * B_M



def r0_from_density(rho_gcc: float, A: float) -> float:
    """
    Compute cell radius r0 from mass density.

    Invert rho = A * m_p / v,  v = (4/3)*pi*r0^3:
        r0 = ( A * m_p / ( (4/3)*pi*rho ) )^(1/3)

    Args:
        rho_gcc: mass density [g/cm^3]
        A:       atomic mass number (e.g. 27 for Al)

    Returns:
        r0: cell radius [m]
    """
    rho_si = rho_gcc * 1e3                          # [kg/m^3]
    volume = A * M_PROTON / rho_si                  # [m^3/atom]
    return (volume * 3.0 / (4.0 * math.pi))**(1.0 / 3.0)


def density_from_r0(r0_m: float, A: float) -> float:
    """
    Compute mass density from cell radius.

    rho = A * m_p / ( (4/3)*pi*r0^3 )

    Args:
        r0_m: cell radius [m]
        A:    atomic mass number

    Returns:
        rho: mass density [g/cm^3]
    """
    volume = (4.0 / 3.0) * math.pi * r0_m**3       # [m^3]
    rho_si = A * M_PROTON / volume                  # [kg/m^3]
    return rho_si * 1e-3                            # [g/cm^3]