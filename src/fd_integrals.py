"""
Fermi-Dirac integral wrapper for the Thomas-Fermi PINN.

This module provides access to Fermi-Dirac integrals (F_{-1/2}, F_{1/2}, F_{3/2})
using Fukushima's rational approximations, implemented in PyTorch with custom
backward passes for automatic differentiation.

Source: MinimalTFFDintPy repository by Aidan Crilly
Location: /rds/general/user/gs1622/home/MinimalTFFDintPy

The Fermi-Dirac integrals from MinimalTFFDintPy include the standard Gamma function 
normalization (division by Gamma(k+1)):
    - F_{-1/2}(η) / Γ(1/2) = F_{-1/2}(η) / √π
    - F_{1/2}(η) / Γ(3/2) = F_{1/2}(η) / (0.5√π)  
    - F_{3/2}(η) / Γ(5/2) = F_{3/2}(η) / (0.75√π)

Physical constants for the temperature-dependent Thomas-Fermi equation:
    - c0 = 1.602e-9 cm = 1.602e-11 m (temperature-dependent length scale coefficient)
    - γ = 0.0899 * Z / T_kV^(3/4)  (reduced potential at origin, Z=1 for hydrogen)
    - λ = α * b * T_kV^(1/4) / c0   (dimensionless length scale ratio)
"""
import sys
import math

# Physical constants
C0_CM = 1.602e-9      # cm, from FMT Eq. 16
C0_M = 1.602e-11      # meters
A0_M = 5.291772105e-11  # Bohr radius in meters
B_M = 0.25 * (4.5 * math.pi**2)**(1/3) * A0_M  # Our length scale b in meters

# Add the external FDint repo to path
FDINT_PATH = '/rds/general/user/gs1622/home/MinimalTFFDintPy'
if FDINT_PATH not in sys.path:
    sys.path.insert(0, FDINT_PATH)

import torch

# Import the public API from MinimalTFFDintPy (with Gamma normalization included)
from FDint_PyTorch import (
    fermi_dirac_integral_minus_half,
    fermi_dirac_integral_half,
    fermi_dirac_integral_three_half,
)

# Re-export with simpler names for use in this project
# Need to ignore gamma normalization in the end
def fermi_dirac_minus_half(x):
    return fermi_dirac_integral_minus_half(x) * (math.sqrt(math.pi))

def fermi_dirac_half(x):
    return fermi_dirac_integral_half(x) * (0.5 * math.sqrt(math.pi))

def fermi_dirac_three_half(x):
    return fermi_dirac_integral_three_half(x) * (0.75 * math.sqrt(math.pi))


def compute_lambda(alpha, T_kV):
    """
    Compute λ = α * b * T_kV^(1/4) / c0

    Args:
        alpha: Tensor of α = r0/b values
        T_kV: Tensor of temperature values in keV

    Returns:
        λ tensor (same shape as inputs after broadcasting)
    """
    return alpha * B_M * (T_kV ** 0.25) / C0_M


def compute_gamma(T_kV, Z=1):
    """
    Compute γ = 0.0899 * Z / T_kV^(3/4)

    Args:
        T_kV: Tensor of temperature values in keV
        Z: Atomic number (default 1 for hydrogen)

    Returns:
        γ tensor (same shape as T_kV)
    """
    return 0.0899 * Z / (T_kV ** 0.75)
