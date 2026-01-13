"""
Extrinsic Curvature for Cylindrical Hypersurfaces

Implements the extrinsic curvature (second fundamental form) K_ab for
ρ = const hypersurfaces in the Einstein-Rosen metric.

The extrinsic curvature measures how the hypersurface bends within
the ambient spacetime:
    K_ab = -n_μ;ν e^μ_a e^ν_b = -½ ℒ_n γ_ab

For the Einstein-Rosen metric, the non-zero components are:
    K_ΦΦ = R e^(-μ+(μ-λ)/2) (1 - ½R μ_ρ)
    K_zz = ½ e^((μ+λ)/2) μ_ρ
    K_tt = ½ e^((λ-μ)/2) (λ_ρ - μ_ρ)
"""

import numpy as np
from typing import Dict, Optional

from .hypersurface import CylindricalHypersurface


class ExtrinsicCurvature:
    """
    Computes extrinsic curvature K_ab for a cylindrical hypersurface.

    The extrinsic curvature (second fundamental form) describes how
    the hypersurface is embedded in the 4D spacetime. It is essential
    for the Israel-Darmois junction conditions.
    """

    def __init__(self, hypersurface: CylindricalHypersurface, solution,
                 eps: float = 1e-6):
        """
        Initialize extrinsic curvature calculator.

        Args:
            hypersurface: The CylindricalHypersurface at ρ = R
            solution: A solution object with mu() and lambda_() methods
            eps: Step size for numerical differentiation
        """
        self.hypersurface = hypersurface
        self.solution = solution
        self.eps = eps
        self.R = hypersurface.radius

    def _metric_derivatives(self, t: float) -> Dict[str, float]:
        """
        Compute radial derivatives of metric functions at the boundary.

        Uses central difference when possible, falls back to one-sided.

        Args:
            t: Time coordinate

        Returns:
            Dictionary with 'mu', 'lambda', 'mu_rho', 'lambda_rho'
        """
        R = self.R
        eps = self.eps

        # Values at boundary
        mu = self.solution.mu(R, t)
        lam = self.solution.lambda_(R, t)

        # One-sided derivatives from interior
        mu_minus = self.solution.mu(R - eps, t)
        lam_minus = self.solution.lambda_(R - eps, t)

        # Try central difference
        try:
            mu_plus = self.solution.mu(R + eps, t)
            lam_plus = self.solution.lambda_(R + eps, t)

            mu_rho = (mu_plus - mu_minus) / (2 * eps)
            lam_rho = (lam_plus - lam_minus) / (2 * eps)
        except Exception:
            # Fall back to one-sided from interior
            mu_rho = (mu - mu_minus) / eps
            lam_rho = (lam - lam_minus) / eps

        return {
            'mu': mu,
            'lambda': lam,
            'mu_rho': mu_rho,
            'lambda_rho': lam_rho,
        }

    def K_ab(self, t: float) -> np.ndarray:
        """
        Compute the extrinsic curvature tensor K_ab.

        The extrinsic curvature is computed from the Lie derivative of
        the induced metric along the normal:
            K_ab = -½ ℒ_n γ_ab

        For our diagonal metric, this simplifies to:
            K_ab = -½ n^ρ ∂γ_ab/∂ρ

        With n^ρ = e^((μ-λ)/2) (outward pointing), we get:
            K_ΦΦ = R e^(-μ+(μ-λ)/2) (1 - ½R μ_ρ)
            K_zz = ½ e^((μ+λ)/2) μ_ρ
            K_tt = ½ e^((λ-μ)/2) (λ_ρ - μ_ρ)

        Args:
            t: Time coordinate

        Returns:
            3x3 matrix K_ab in coordinates (Φ, z, t)
        """
        R = self.R
        derivs = self._metric_derivatives(t)

        mu = derivs['mu']
        lam = derivs['lambda']
        mu_rho = derivs['mu_rho']
        lam_rho = derivs['lambda_rho']

        # Compute each component
        # K_ΦΦ: derivative of γ_ΦΦ = -R²e^(-μ)
        # ∂γ_ΦΦ/∂ρ = -2R e^(-μ) + R² e^(-μ) μ_ρ = -e^(-μ)(2R - R²μ_ρ)
        # K_ΦΦ = -½ n^ρ ∂γ_ΦΦ/∂ρ = ½ e^((μ-λ)/2) e^(-μ)(2R - R²μ_ρ)
        #      = ½ e^(-μ+(μ-λ)/2) (2R - R²μ_ρ)
        #      = R e^((-μ-λ)/2) (1 - ½R μ_ρ)
        K_PhiPhi = R * np.exp((-mu - lam) / 2) * (1 - 0.5 * R * mu_rho)

        # K_zz: derivative of γ_zz = -e^μ
        # ∂γ_zz/∂ρ = -e^μ μ_ρ
        # K_zz = -½ n^ρ ∂γ_zz/∂ρ = ½ e^((μ-λ)/2) e^μ μ_ρ
        #      = ½ e^((3μ-λ)/2) μ_ρ
        K_zz = 0.5 * np.exp((3 * mu - lam) / 2) * mu_rho

        # K_tt: derivative of γ_tt = e^(λ-μ)
        # ∂γ_tt/∂ρ = e^(λ-μ) (λ_ρ - μ_ρ)
        # K_tt = -½ n^ρ ∂γ_tt/∂ρ = -½ e^((μ-λ)/2) e^(λ-μ) (λ_ρ - μ_ρ)
        #      = -½ e^((λ-μ)/2) (λ_ρ - μ_ρ)
        K_tt = -0.5 * np.exp((lam - mu) / 2) * (lam_rho - mu_rho)

        return np.array([
            [K_PhiPhi, 0.0, 0.0],
            [0.0, K_zz, 0.0],
            [0.0, 0.0, K_tt]
        ])

    def K_trace(self, t: float) -> float:
        """
        Compute the trace of extrinsic curvature K = γ^ab K_ab.

        This is also called the mean curvature (up to a factor).

        Args:
            t: Time coordinate

        Returns:
            Trace K
        """
        K = self.K_ab(t)
        gamma_inv = self.hypersurface.induced_metric_inverse(self.solution, t)

        # K = γ^ab K_ab (sum over diagonal for diagonal tensors)
        trace = np.sum(gamma_inv * K)

        return trace

    def K_ab_traceless(self, t: float) -> np.ndarray:
        """
        Compute the traceless part of extrinsic curvature.

        K̃_ab = K_ab - ⅓ γ_ab K

        This is sometimes useful for analyzing shear.

        Args:
            t: Time coordinate

        Returns:
            3x3 traceless extrinsic curvature
        """
        K = self.K_ab(t)
        K_tr = self.K_trace(t)
        gamma = self.hypersurface.induced_metric(self.solution, t)

        return K - (1.0 / 3.0) * gamma * K_tr

    def K_squared(self, t: float) -> float:
        """
        Compute K_ab K^ab = γ^ac γ^bd K_ab K_cd.

        This quantity appears in the Gauss-Codazzi equations.

        Args:
            t: Time coordinate

        Returns:
            K_ab K^ab
        """
        K = self.K_ab(t)
        gamma_inv = self.hypersurface.induced_metric_inverse(self.solution, t)

        # For diagonal tensors: K^ab = γ^ac γ^bd K_cd = γ^aa γ^bb K_ab
        # K_ab K^ab = sum_a sum_b K_ab * γ^aa * γ^bb * K_ab
        K_sq = 0.0
        for a in range(3):
            for b in range(3):
                K_sq += K[a, b] * gamma_inv[a, a] * gamma_inv[b, b] * K[a, b]

        return K_sq

    def principal_curvatures(self, t: float) -> np.ndarray:
        """
        Compute principal curvatures (eigenvalues of K^a_b).

        The principal curvatures are the eigenvalues of the mixed
        tensor K^a_b = γ^ac K_cb.

        Args:
            t: Time coordinate

        Returns:
            Array of 3 principal curvatures
        """
        K = self.K_ab(t)
        gamma_inv = self.hypersurface.induced_metric_inverse(self.solution, t)

        # Raise first index: K^a_b = γ^ac K_cb
        K_mixed = gamma_inv @ K

        # Eigenvalues
        eigenvalues = np.linalg.eigvals(K_mixed)

        return np.real(eigenvalues)

    def gaussian_curvature_contribution(self, t: float) -> float:
        """
        Compute the contribution to Gaussian curvature from extrinsic curvature.

        From the Gauss equation:
        R(3) = R(4) - 2 R_μν n^μ n^ν + K² - K_ab K^ab

        This returns K² - K_ab K^ab, the extrinsic contribution.

        Args:
            t: Time coordinate

        Returns:
            K² - K_ab K^ab
        """
        K_tr = self.K_trace(t)
        K_sq = self.K_squared(t)

        return K_tr ** 2 - K_sq

    def summary(self, t: float) -> Dict[str, float]:
        """
        Get a summary of all extrinsic curvature quantities.

        Args:
            t: Time coordinate

        Returns:
            Dictionary with all computed quantities
        """
        K = self.K_ab(t)
        principal = self.principal_curvatures(t)

        return {
            'K_PhiPhi': K[0, 0],
            'K_zz': K[1, 1],
            'K_tt': K[2, 2],
            'K_trace': self.K_trace(t),
            'K_squared': self.K_squared(t),
            'principal_1': principal[0],
            'principal_2': principal[1],
            'principal_3': principal[2],
            'gauss_contribution': self.gaussian_curvature_contribution(t),
        }

    def __repr__(self) -> str:
        return f"ExtrinsicCurvature(R={self.R}, solution={type(self.solution).__name__})"
