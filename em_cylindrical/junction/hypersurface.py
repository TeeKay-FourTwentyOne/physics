"""
Cylindrical Hypersurface Geometry

Implements the geometry of ρ = const hypersurfaces in the Einstein-Rosen
metric, including the induced metric (first fundamental form) and
normal vectors.

For the Einstein-Rosen metric:
    ds² = e^(λ-μ)(dt² - dρ²) - ρ²e^(-μ)dΦ² - e^μdz²

A hypersurface at ρ = R has:
- Tangent coordinates: (Φ, z, t)
- Normal direction: ρ
- Induced metric: γ_ab = diag(-R²e^(-μ), -e^μ, e^(λ-μ))
"""

import numpy as np
from typing import Tuple, List


class CylindricalHypersurface:
    """
    Represents a ρ = R hypersurface for junction conditions.

    The hypersurface is a 3D timelike surface embedded in the 4D spacetime.
    It has coordinates (Φ, z, t) and normal direction ρ.
    """

    def __init__(self, radius: float):
        """
        Initialize hypersurface at given radius.

        Args:
            radius: The radial coordinate R where the hypersurface is located.
                    Must be positive.
        """
        if radius <= 0:
            raise ValueError(f"Radius must be positive, got {radius}")
        self.radius = radius
        self.R = radius  # Alias for convenience

    def tangent_coordinates(self) -> List[str]:
        """
        Get the coordinate names on the hypersurface.

        Returns:
            List of coordinate names: ['Phi', 'z', 't']
        """
        return ['Phi', 'z', 't']

    def normal_covector(self, solution, t: float) -> np.ndarray:
        """
        Compute the unit normal covector n_μ (covariant, pointing outward).

        For ρ = const surface, the normal is in the ρ direction.
        The unit normal covector is: n_ρ = √|g_ρρ|

        For Einstein-Rosen metric: g_ρρ = -e^(λ-μ) (spacelike)
        So |g_ρρ| = e^(λ-μ), and n_ρ = e^((λ-μ)/2)

        Args:
            solution: A solution object with mu() and lambda_() methods
            t: Time coordinate

        Returns:
            4-component array [n_ρ, n_Φ, n_z, n_t]
        """
        R = self.radius
        mu = solution.mu(R, t)
        lam = solution.lambda_(R, t)

        # Unit normal magnitude
        n_rho = np.exp((lam - mu) / 2)

        return np.array([n_rho, 0.0, 0.0, 0.0])

    def normal_vector(self, solution, t: float) -> np.ndarray:
        """
        Compute the unit normal vector n^μ (contravariant, pointing outward).

        n^μ = g^μν n_ν

        For the diagonal metric:
        n^ρ = g^ρρ n_ρ = -e^(-(λ-μ)) * e^((λ-μ)/2) = -e^(-(λ-μ)/2)

        The negative sign indicates outward direction for increasing ρ.
        We use the convention that outward normal has n^ρ > 0, so:
        n^ρ = e^((μ-λ)/2)

        Args:
            solution: A solution object with mu() and lambda_() methods
            t: Time coordinate

        Returns:
            4-component array [n^ρ, n^Φ, n^z, n^t]
        """
        R = self.radius
        mu = solution.mu(R, t)
        lam = solution.lambda_(R, t)

        # Contravariant component (outward pointing)
        # g^ρρ = -e^(-(λ-μ)) = -e^(μ-λ)
        # n^ρ = g^ρρ * n_ρ = -e^(μ-λ) * e^((λ-μ)/2) = -e^((μ-λ)/2)
        # For outward normal, we want positive, so use sign convention
        n_rho_up = np.exp((mu - lam) / 2)

        return np.array([n_rho_up, 0.0, 0.0, 0.0])

    def induced_metric(self, solution, t: float) -> np.ndarray:
        """
        Compute the induced metric (first fundamental form) γ_ab.

        The induced metric is the restriction of the spacetime metric
        to the hypersurface. For coordinates (Φ, z, t):

        γ_ΦΦ = g_ΦΦ = -R² e^(-μ)
        γ_zz = g_zz = -e^μ
        γ_tt = g_tt = e^(λ-μ)

        Args:
            solution: A solution object with mu() and lambda_() methods
            t: Time coordinate

        Returns:
            3x3 matrix γ_ab in coordinates (Φ, z, t)
        """
        R = self.radius
        mu = solution.mu(R, t)
        lam = solution.lambda_(R, t)

        gamma_PhiPhi = -R**2 * np.exp(-mu)
        gamma_zz = -np.exp(mu)
        gamma_tt = np.exp(lam - mu)

        return np.array([
            [gamma_PhiPhi, 0.0, 0.0],
            [0.0, gamma_zz, 0.0],
            [0.0, 0.0, gamma_tt]
        ])

    def induced_metric_inverse(self, solution, t: float) -> np.ndarray:
        """
        Compute the inverse induced metric γ^ab.

        For diagonal metric:
        γ^ΦΦ = 1/γ_ΦΦ = -e^μ / R²
        γ^zz = 1/γ_zz = -e^(-μ)
        γ^tt = 1/γ_tt = e^(μ-λ)

        Args:
            solution: A solution object with mu() and lambda_() methods
            t: Time coordinate

        Returns:
            3x3 matrix γ^ab in coordinates (Φ, z, t)
        """
        R = self.radius
        mu = solution.mu(R, t)
        lam = solution.lambda_(R, t)

        gamma_inv_PhiPhi = -np.exp(mu) / R**2
        gamma_inv_zz = -np.exp(-mu)
        gamma_inv_tt = np.exp(mu - lam)

        return np.array([
            [gamma_inv_PhiPhi, 0.0, 0.0],
            [0.0, gamma_inv_zz, 0.0],
            [0.0, 0.0, gamma_inv_tt]
        ])

    def induced_metric_determinant(self, solution, t: float) -> float:
        """
        Compute the determinant of the induced metric.

        det(γ) = γ_ΦΦ * γ_zz * γ_tt
               = (-R² e^(-μ)) * (-e^μ) * (e^(λ-μ))
               = R² e^(λ-μ)

        Args:
            solution: A solution object with mu() and lambda_() methods
            t: Time coordinate

        Returns:
            Determinant of γ_ab
        """
        R = self.radius
        mu = solution.mu(R, t)
        lam = solution.lambda_(R, t)

        return R**2 * np.exp(lam - mu)

    def metric_derivatives_at_boundary(self, solution, t: float,
                                       eps: float = 1e-6) -> dict:
        """
        Compute radial derivatives of metric functions at the boundary.

        Returns ∂μ/∂ρ and ∂λ/∂ρ evaluated at ρ = R.

        Args:
            solution: A solution object
            t: Time coordinate
            eps: Step size for numerical differentiation

        Returns:
            Dictionary with 'mu_rho' and 'lambda_rho'
        """
        R = self.radius

        # One-sided derivative from interior (R - eps)
        mu_minus = solution.mu(R - eps, t)
        mu_center = solution.mu(R, t)

        lam_minus = solution.lambda_(R - eps, t)
        lam_center = solution.lambda_(R, t)

        # Use central difference if possible, else one-sided
        try:
            mu_plus = solution.mu(R + eps, t)
            lam_plus = solution.lambda_(R + eps, t)

            mu_rho = (mu_plus - mu_minus) / (2 * eps)
            lam_rho = (lam_plus - lam_minus) / (2 * eps)
        except Exception:
            # Fall back to one-sided
            mu_rho = (mu_center - mu_minus) / eps
            lam_rho = (lam_center - lam_minus) / eps

        return {
            'mu_rho': mu_rho,
            'lambda_rho': lam_rho,
        }

    def area_element(self, solution, t: float) -> float:
        """
        Compute the area element √|det(γ)| on the hypersurface.

        √|det(γ)| = R * e^((λ-μ)/2)

        Args:
            solution: A solution object
            t: Time coordinate

        Returns:
            Area element per unit coordinate volume
        """
        R = self.radius
        mu = solution.mu(R, t)
        lam = solution.lambda_(R, t)

        return R * np.exp((lam - mu) / 2)

    def __repr__(self) -> str:
        return f"CylindricalHypersurface(radius={self.radius})"
