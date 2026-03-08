"""
Electron chemical potential for the Thomas-Fermi EoS.

Source: QEOS (More et al. 1988) gauge choice stated below Eq. 69: V(R0) = 0.
With this gauge the FD argument at the boundary is exactly mu_e / kT:

    xi(1) = gamma * phi(1) / lambda = mu_e,1 / (kT_1)

So the Z=1 chemical potential is simply:

    mu_e1 = xi(1) * kT_1 = xi(1) * T_1_keV * KEV_TO_J    [J]

Z-scaling: from FMT Section VI similarity transformation, energy per electron
scales as Z^(4/3), therefore:

    mu_e = Z^(4/3) * mu_e1    [J]

This is a boundary-only quantity — no volume integral required.
mu_e can be negative (classical/non-degenerate regime, large r0 or high T).
"""

from ..inputs import KEV_TO_J


def compute_chemical_potential(
    phi_boundary: float,
    T_1_keV: float,
    gamma: float,
    lam: float,
    Z: float,
) -> dict:
    """
    Compute electron chemical potential in Z=1 system and scale to physical Z.

    Args:
        phi_boundary: phi(x=1) from PINN [dimensionless]
        T_1_keV:      Z=1 reduced temperature [keV]
        gamma:        0.0899 / T_1_keV^(3/4) [dimensionless]
        lam:          alpha_1 * b * T_1_keV^(1/4) / C0 [dimensionless]
        Z:            atomic number of physical element

    Returns:
        dict with keys:
            'xi_1'   : FD argument at boundary = mu_e1 / kT_1  [dimensionless]
            'mu_e1'  : Z=1 electron chemical potential          [J]
            'mu_e'   : physical electron chemical potential     [J]
            'mu_e_keV: physical electron chemical potential     [keV]
    """
    # xi(1) = mu_e1 / kT_1  (QEOS gauge V(R0)=0, below Eq. 69)
    xi_1 = gamma * phi_boundary / lam

    # Z=1 chemical potential [J]
    mu_e1 = xi_1 * T_1_keV * KEV_TO_J

    # Scale to physical Z (FMT Section VI: energy per electron ~ Z^(4/3))
    mu_e = Z**(4.0 / 3.0) * mu_e1

    return {
        "xi_1":     xi_1,
        "mu_e1":    mu_e1,
        "mu_e":     mu_e,
        "mu_e_keV": mu_e / KEV_TO_J,
    }