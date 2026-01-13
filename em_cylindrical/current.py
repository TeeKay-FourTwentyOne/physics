"""
Current Density Calculations

Computes the 4-current density J^μ from Maxwell's equations in curved spacetime:
    ∇_μ F^μν = (1/√(-g)) ∂_μ (√(-g) F^μν) = 4π J^ν

For vacuum solutions (in vacuo), J^μ = 0 everywhere.
This module verifies that property and provides tools for computing currents
when sources are present.
"""

import sympy as sp
import numpy as np
from typing import Dict, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .solutions.case_i import CaseISolution
    from .solutions.case_ii import CaseIISolution
    from .solutions.case_iii import CaseIIISolution

# Type alias for any solution
Solution = Union['CaseISolution', 'CaseIISolution', 'CaseIIISolution']


class CurrentDensity:
    """
    Compute 4-current density J^μ from Maxwell's equations.

    For the Einstein-Rosen metric:
        ds² = e^(λ-μ)(dt² - dρ²) - ρ²e^(-μ)dΦ² - e^μdz²

    The metric determinant is:
        g = det(g_μν) = -ρ² e^(2λ-2μ)
        √(-g) = ρ e^(λ-μ)

    Maxwell's equations with sources:
        ∇_μ F^μν = 4π J^ν

    Coordinates: (ρ, Φ, z, t) = indices (0, 1, 2, 3)
    """

    def __init__(self, solution: Solution):
        """
        Initialize with a solution object.

        Args:
            solution: A CaseISolution, CaseIISolution, or CaseIIISolution
        """
        self.solution = solution

    def _sqrt_neg_g(self, rho_val: float, t_val: float) -> float:
        """
        Compute √(-g) = ρ e^(λ-μ) at a point.
        """
        lam = self.solution.lambda_(rho_val, t_val)
        mu = self.solution.mu(rho_val, t_val)
        return rho_val * np.exp(lam - mu)

    def _inverse_metric(self, rho_val: float, t_val: float) -> np.ndarray:
        """
        Compute inverse metric g^μν at a point.

        For the diagonal Einstein-Rosen metric:
            g^ρρ = -e^(μ-λ)
            g^ΦΦ = -e^μ / ρ²
            g^zz = -e^(-μ)
            g^tt = e^(μ-λ)
        """
        lam = self.solution.lambda_(rho_val, t_val)
        mu = self.solution.mu(rho_val, t_val)

        exp_mu_minus_lam = np.exp(mu - lam)
        exp_mu = np.exp(mu)
        exp_minus_mu = np.exp(-mu)

        return np.array([
            [-exp_mu_minus_lam, 0, 0, 0],
            [0, -exp_mu / rho_val**2, 0, 0],
            [0, 0, -exp_minus_mu, 0],
            [0, 0, 0, exp_mu_minus_lam]
        ])

    def _F_upper(self, rho_val: float, t_val: float) -> np.ndarray:
        """
        Compute F^μν (contravariant field tensor) by raising indices.

        F^μν = g^μα g^νβ F_αβ
        """
        F_lower = self.solution.field_tensor_at(rho_val, t_val)
        g_inv = self._inverse_metric(rho_val, t_val)

        # F^μν = g^μα F_αβ g^βν
        # Since g is diagonal, this simplifies to F^μν = g^μμ g^νν F_μν (no sum)
        F_upper = np.zeros((4, 4))
        for mu in range(4):
            for nu in range(4):
                F_upper[mu, nu] = g_inv[mu, mu] * g_inv[nu, nu] * F_lower[mu, nu]

        return F_upper

    def divergence_F(self, rho_val: float, t_val: float) -> np.ndarray:
        """
        Compute the covariant divergence ∇_μ F^μν = 4π J^ν.

        Uses: ∇_μ F^μν = (1/√(-g)) ∂_μ (√(-g) F^μν)

        Returns:
            4-vector (4π J^ρ, 4π J^Φ, 4π J^z, 4π J^t)
        """
        eps = 1e-6
        sqrt_g = self._sqrt_neg_g(rho_val, t_val)

        div_F = np.zeros(4)

        for nu in range(4):
            # Compute ∂_ρ (√(-g) F^ρν)
            sqrt_g_F_rho_nu_plus = self._sqrt_neg_g(rho_val + eps, t_val) * \
                                   self._F_upper(rho_val + eps, t_val)[0, nu]
            sqrt_g_F_rho_nu_minus = self._sqrt_neg_g(rho_val - eps, t_val) * \
                                    self._F_upper(rho_val - eps, t_val)[0, nu]
            d_rho = (sqrt_g_F_rho_nu_plus - sqrt_g_F_rho_nu_minus) / (2 * eps)

            # Compute ∂_t (√(-g) F^tν)
            sqrt_g_F_t_nu_plus = self._sqrt_neg_g(rho_val, t_val + eps) * \
                                  self._F_upper(rho_val, t_val + eps)[3, nu]
            sqrt_g_F_t_nu_minus = self._sqrt_neg_g(rho_val, t_val - eps) * \
                                   self._F_upper(rho_val, t_val - eps)[3, nu]
            d_t = (sqrt_g_F_t_nu_plus - sqrt_g_F_t_nu_minus) / (2 * eps)

            # Note: ∂_Φ and ∂_z terms are zero due to cylindrical symmetry
            # (fields don't depend on Φ or z)

            div_F[nu] = (d_rho + d_t) / sqrt_g

        return div_F

    def current_4vector_at(self, rho_val: float, t_val: float) -> np.ndarray:
        """
        Compute the 4-current density J^μ at a point.

        Returns:
            J^μ = (J^ρ, J^Φ, J^z, J^t) where J^t is charge density times c
        """
        div_F = self.divergence_F(rho_val, t_val)
        return div_F / (4 * np.pi)

    def verify_vacuum(self, rho_val: float, t_val: float) -> Dict[str, float]:
        """
        Verify that J^μ ≈ 0 for vacuum solutions.

        Returns dictionary with residuals for each component of J^μ.
        These should all be close to zero for valid vacuum solutions.
        """
        J = self.current_4vector_at(rho_val, t_val)

        return {
            'J_rho': J[0],
            'J_Phi': J[1],
            'J_z': J[2],
            'J_t': J[3],
            'J_magnitude': np.sqrt(np.sum(J**2)),
        }

    def charge_density(self, rho_val: float, t_val: float) -> float:
        """
        Compute the charge density ρ_charge = J^t (in geometric units).

        For vacuum solutions, this should be ~0.
        """
        J = self.current_4vector_at(rho_val, t_val)
        return J[3]

    def current_density_z(self, rho_val: float, t_val: float) -> float:
        """
        Compute the axial current density J^z.

        For vacuum solutions, this should be ~0.
        """
        J = self.current_4vector_at(rho_val, t_val)
        return J[2]

    def verify_vacuum_grid(self, rho_range: tuple, t_range: tuple,
                           n_points: int = 5) -> Dict[str, float]:
        """
        Verify J ≈ 0 over a grid of points.

        Args:
            rho_range: (rho_min, rho_max)
            t_range: (t_min, t_max)
            n_points: Number of points in each dimension

        Returns:
            Dictionary with max |J| components over the grid
        """
        rho_vals = np.linspace(rho_range[0], rho_range[1], n_points)
        t_vals = np.linspace(t_range[0], t_range[1], n_points)

        max_J = np.zeros(4)

        for rho_val in rho_vals:
            for t_val in t_vals:
                J = self.current_4vector_at(rho_val, t_val)
                max_J = np.maximum(max_J, np.abs(J))

        return {
            'max_J_rho': max_J[0],
            'max_J_Phi': max_J[1],
            'max_J_z': max_J[2],
            'max_J_t': max_J[3],
            'max_J_magnitude': np.sqrt(np.sum(max_J**2)),
        }
