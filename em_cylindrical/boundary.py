"""
Surface Current Calculations at Cylindrical Boundaries

Computes surface currents K^μ required at a cylindrical boundary ρ = R
to sustain the electromagnetic field.

At a boundary, the jump in the tangential magnetic field requires a surface current:
    K^μ = (1/4π) n_α [F^αμ]  (jump condition)

where n_α is the outward normal and [F^αμ] is the discontinuity.

For a field that exists only inside ρ < R (with vacuum outside), the required
surface current is:
    K^μ = -(1/4π) n_α F^αμ|_{ρ=R}
"""

import numpy as np
from typing import Dict, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .solutions.case_i import CaseISolution
    from .solutions.case_ii import CaseIISolution
    from .solutions.case_iii import CaseIIISolution

Solution = Union['CaseISolution', 'CaseIISolution', 'CaseIIISolution']


class CylindricalBoundary:
    """
    Compute surface currents at a cylindrical boundary ρ = R.

    Physical interpretation:
    - For Case II (ψ=0, φ≠0): φ relates to azimuthal potential
        - F₁₂ = -∂φ/∂ρ gives radial-azimuthal field component
        - Surface currents flow in z and Φ directions
    - For Case I (φ=0, ψ≠0): ψ relates to axial potential
        - F₁₃ = -∂ψ/∂ρ gives radial-axial field component
        - Surface currents flow in z and Φ directions

    Coordinates: (ρ, Φ, z, t) = indices (0, 1, 2, 3)
    """

    def __init__(self, solution: Solution, radius: float):
        """
        Initialize boundary at ρ = radius.

        Args:
            solution: A solution object (CaseI, CaseII, or CaseIII)
            radius: Boundary radius R
        """
        self.solution = solution
        self.radius = radius

    def _inverse_metric_at_boundary(self, t_val: float) -> np.ndarray:
        """Compute inverse metric g^μν at the boundary."""
        lam = self.solution.lambda_(self.radius, t_val)
        mu = self.solution.mu(self.radius, t_val)

        exp_mu_minus_lam = np.exp(mu - lam)
        exp_mu = np.exp(mu)
        exp_minus_mu = np.exp(-mu)

        return np.array([
            [-exp_mu_minus_lam, 0, 0, 0],
            [0, -exp_mu / self.radius**2, 0, 0],
            [0, 0, -exp_minus_mu, 0],
            [0, 0, 0, exp_mu_minus_lam]
        ])

    def _F_upper_at_boundary(self, t_val: float) -> np.ndarray:
        """Compute F^μν at the boundary."""
        F_lower = self.solution.field_tensor_at(self.radius, t_val)
        g_inv = self._inverse_metric_at_boundary(t_val)

        F_upper = np.zeros((4, 4))
        for mu in range(4):
            for nu in range(4):
                F_upper[mu, nu] = g_inv[mu, mu] * g_inv[nu, nu] * F_lower[mu, nu]

        return F_upper

    def _normal_covariant(self, t_val: float) -> np.ndarray:
        """
        Compute the outward-pointing unit normal n_α (covariant).

        For a surface at constant ρ, the normal is in the ρ direction.
        The covariant normal is n_α = (n_ρ, 0, 0, 0) where:
            n_ρ = √|g_ρρ| = √(e^(λ-μ)) for proper normalization
        """
        lam = self.solution.lambda_(self.radius, t_val)
        mu = self.solution.mu(self.radius, t_val)

        # g_ρρ = -e^(λ-μ), so |g_ρρ| = e^(λ-μ)
        n_rho = np.sqrt(np.exp(lam - mu))

        return np.array([n_rho, 0, 0, 0])

    def surface_current_K(self, t_val: float) -> np.ndarray:
        """
        Compute surface current density K^μ at the boundary.

        For a field existing only inside ρ < R with vacuum outside:
            K^μ = -(1/4π) n_α F^αμ|_{ρ=R}

        Returns:
            K^μ = (K^ρ, K^Φ, K^z, K^t)
            Note: K^ρ should be ~0 (no current normal to surface)
        """
        F_upper = self._F_upper_at_boundary(t_val)
        n = self._normal_covariant(t_val)

        # K^μ = -(1/4π) n_α F^αμ = -(1/4π) n_ρ F^ρμ (only ρ component of n is non-zero)
        K = np.zeros(4)
        for mu in range(4):
            K[mu] = -(1 / (4 * np.pi)) * n[0] * F_upper[0, mu]

        return K

    def surface_current_physical(self, t_val: float) -> Dict[str, float]:
        """
        Return surface current components with physical interpretation.

        Returns dictionary with:
            K_z: Axial current density (current per unit azimuthal length)
            K_Phi: Azimuthal current density (current per unit axial length)
            K_t: Charge density on surface
        """
        K = self.surface_current_K(t_val)

        return {
            'K_rho': K[0],   # Should be ~0 (tangent to surface)
            'K_Phi': K[1],   # Azimuthal current
            'K_z': K[2],     # Axial current
            'K_t': K[3],     # Surface charge density
        }

    def total_current_z(self, t_val: float) -> float:
        """
        Compute total axial current I_z through the cylinder.

        I_z = ∮ K^z R dΦ = 2π R K^z

        Returns:
            Total axial current (in geometric units)
        """
        K = self.surface_current_K(t_val)
        K_z = K[2]

        # Integrate around circumference: I = K^z * circumference
        # But K^z is already current per unit circumference, so:
        return 2 * np.pi * self.radius * K_z

    def total_current_Phi(self, z_length: float, t_val: float) -> float:
        """
        Compute total azimuthal current over a length z_length.

        I_Φ = ∫ K^Φ dz = K^Φ * z_length

        Args:
            z_length: Length of cylinder segment
            t_val: Time

        Returns:
            Total azimuthal current
        """
        K = self.surface_current_K(t_val)
        K_Phi = K[1]

        return K_Phi * z_length

    def magnetic_field_at_boundary(self, t_val: float) -> Dict[str, float]:
        """
        Extract physical magnetic field components at the boundary.

        For the cylindrically symmetric case:
            B_z ∝ F₁₂ (from ∂φ/∂ρ)
            B_Φ ∝ F₁₃ (from ∂ψ/∂ρ)

        Returns dictionary with field components.
        """
        F = self.solution.field_tensor_at(self.radius, t_val)

        # Field components (not properly normalized for curved spacetime)
        return {
            'F_12': F[0, 1],  # Related to B_z
            'F_13': F[0, 2],  # Related to B_Phi
            'F_24': F[1, 3],  # Related to E_Phi
            'F_34': F[2, 3],  # Related to E_z
        }

    def amperes_law_check(self, t_val: float) -> Dict[str, float]:
        """
        Verify consistency with Ampère's law.

        For a current-carrying cylinder, Ampère's law relates:
            ∮ B · dl = (4π/c) I_enclosed

        Returns residuals for this check.
        """
        # Get magnetic field component
        F = self.solution.field_tensor_at(self.radius, t_val)
        F_12 = F[0, 1]  # Related to B_z

        # Get total axial current
        I_z = self.total_current_z(t_val)

        # Ampère's law check (simplified, geometric units)
        # The line integral of B_z around the cylinder should relate to enclosed current
        # This is approximate due to curved spacetime effects
        B_integral = 2 * np.pi * self.radius * F_12
        expected = 4 * np.pi * I_z

        return {
            'B_line_integral': B_integral,
            'expected_from_I': expected,
            'residual': B_integral - expected,
        }


def compute_solenoid_current(solution: Solution, radius: float,
                             t_val: float = 0.0) -> Dict[str, float]:
    """
    Compute the equivalent solenoid current for a given solution.

    This gives the axial current that would produce the magnetic field
    at the boundary.

    Args:
        solution: Electromagnetic solution
        radius: Cylinder radius
        t_val: Time value

    Returns:
        Dictionary with current values
    """
    boundary = CylindricalBoundary(solution, radius)
    K = boundary.surface_current_physical(t_val)
    I_z = boundary.total_current_z(t_val)

    return {
        'surface_current_density_z': K['K_z'],
        'total_axial_current': I_z,
        'current_per_unit_length': K['K_z'],  # For a solenoid
    }
