"""
Einstein-Rosen Metric for Cylindrically Symmetric Spacetimes

The metric (equation 1 from Misra & Radhakrishna 1962):
    ds² = e^(λ-μ)(dt² - dρ²) - ρ²e^(-μ)dΦ² - e^μdz²

Coordinates: (x¹, x², x³, x⁴) = (ρ, Φ, z, t)
"""

import sympy as sp
import numpy as np
from typing import Union, Callable

# Define symbolic variables used throughout
rho, t, Phi, z = sp.symbols('rho t Phi z', real=True, positive=True)
rho_sym, t_sym = rho, t  # Aliases for clarity


class EinsteinRosenMetric:
    """
    Einstein-Rosen metric tensor for cylindrically symmetric spacetimes.

    The metric has the form:
        ds² = e^(λ-μ)(dt² - dρ²) - ρ²e^(-μ)dΦ² - e^μdz²

    where λ and μ are functions of ρ and t only.
    """

    def __init__(self, lambda_expr: sp.Expr, mu_expr: sp.Expr):
        """
        Initialize the metric with symbolic expressions for λ and μ.

        Args:
            lambda_expr: SymPy expression for λ(ρ, t)
            mu_expr: SymPy expression for μ(ρ, t)
        """
        self.lambda_sym = lambda_expr
        self.mu_sym = mu_expr

        # Build symbolic metric tensor
        self._build_metric_tensor()

        # Create numerical evaluation functions
        self._lambda_func = sp.lambdify((rho, t), self.lambda_sym, 'numpy')
        self._mu_func = sp.lambdify((rho, t), self.mu_sym, 'numpy')

    def _build_metric_tensor(self):
        """Construct the 4x4 metric tensor symbolically."""
        lam = self.lambda_sym
        mu = self.mu_sym

        # Metric components g_ij (covariant)
        # Coordinates: (ρ, Φ, z, t) = indices (0, 1, 2, 3)
        # From ds² = e^(λ-μ)(dt² - dρ²) - ρ²e^(-μ)dΦ² - e^μdz²

        exp_lam_minus_mu = sp.exp(lam - mu)
        exp_minus_mu = sp.exp(-mu)
        exp_mu = sp.exp(mu)

        self.g_symbolic = sp.Matrix([
            [-exp_lam_minus_mu, 0, 0, 0],           # g_ρρ, g_ρΦ, g_ρz, g_ρt
            [0, -rho**2 * exp_minus_mu, 0, 0],      # g_Φρ, g_ΦΦ, g_Φz, g_Φt
            [0, 0, -exp_mu, 0],                      # g_zρ, g_zΦ, g_zz, g_zt
            [0, 0, 0, exp_lam_minus_mu]              # g_tρ, g_tΦ, g_tz, g_tt
        ])

        # Also compute inverse metric g^ij (contravariant)
        self.g_inv_symbolic = sp.Matrix([
            [-1/exp_lam_minus_mu, 0, 0, 0],
            [0, -1/(rho**2 * exp_minus_mu), 0, 0],
            [0, 0, -1/exp_mu, 0],
            [0, 0, 0, 1/exp_lam_minus_mu]
        ])

        # Determinant of metric
        self.det_g_symbolic = self.g_symbolic.det()

    def metric_at(self, rho_val: float, t_val: float) -> np.ndarray:
        """
        Evaluate the metric tensor numerically at a point.

        Args:
            rho_val: Radial coordinate value (must be > 0)
            t_val: Time coordinate value

        Returns:
            4x4 numpy array representing g_ij
        """
        lam = self._lambda_func(rho_val, t_val)
        mu = self._mu_func(rho_val, t_val)

        exp_lam_minus_mu = np.exp(lam - mu)
        exp_minus_mu = np.exp(-mu)
        exp_mu = np.exp(mu)

        return np.array([
            [-exp_lam_minus_mu, 0, 0, 0],
            [0, -rho_val**2 * exp_minus_mu, 0, 0],
            [0, 0, -exp_mu, 0],
            [0, 0, 0, exp_lam_minus_mu]
        ])

    def inverse_metric_at(self, rho_val: float, t_val: float) -> np.ndarray:
        """
        Evaluate the inverse metric tensor numerically at a point.

        Args:
            rho_val: Radial coordinate value (must be > 0)
            t_val: Time coordinate value

        Returns:
            4x4 numpy array representing g^ij
        """
        lam = self._lambda_func(rho_val, t_val)
        mu = self._mu_func(rho_val, t_val)

        exp_lam_minus_mu = np.exp(lam - mu)
        exp_minus_mu = np.exp(-mu)
        exp_mu = np.exp(mu)

        return np.array([
            [-1/exp_lam_minus_mu, 0, 0, 0],
            [0, -1/(rho_val**2 * exp_minus_mu), 0, 0],
            [0, 0, -1/exp_mu, 0],
            [0, 0, 0, 1/exp_lam_minus_mu]
        ])

    def ds_squared(self, rho_val: float, t_val: float,
                   drho: float, dPhi: float, dz: float, dt: float) -> float:
        """
        Compute the line element ds² for given coordinate differentials.

        Args:
            rho_val, t_val: Point coordinates
            drho, dPhi, dz, dt: Coordinate differentials

        Returns:
            Value of ds²
        """
        g = self.metric_at(rho_val, t_val)
        dx = np.array([drho, dPhi, dz, dt])
        return float(dx @ g @ dx)

    def lambda_at(self, rho_val: float, t_val: float) -> float:
        """Evaluate λ numerically."""
        return float(self._lambda_func(rho_val, t_val))

    def mu_at(self, rho_val: float, t_val: float) -> float:
        """Evaluate μ numerically."""
        return float(self._mu_func(rho_val, t_val))


def compute_christoffel_symbols(metric: EinsteinRosenMetric) -> dict:
    """
    Compute Christoffel symbols Γ^i_jk symbolically.

    Returns dictionary with keys (i, j, k) mapping to SymPy expressions.
    """
    g = metric.g_symbolic
    g_inv = metric.g_inv_symbolic
    coords = [rho, Phi, z, t]

    christoffel = {}
    for i in range(4):
        for j in range(4):
            for k in range(4):
                gamma = sp.Integer(0)
                for m in range(4):
                    term = g_inv[i, m] * (
                        sp.diff(g[m, j], coords[k]) +
                        sp.diff(g[m, k], coords[j]) -
                        sp.diff(g[j, k], coords[m])
                    )
                    gamma += term
                gamma = sp.Rational(1, 2) * gamma
                gamma = sp.simplify(gamma)
                if gamma != 0:
                    christoffel[(i, j, k)] = gamma

    return christoffel
