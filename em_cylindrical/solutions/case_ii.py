"""
Case (ii): ψ = 0 Solutions

When ψ = 0, only the φ component of the electromagnetic 4-potential is non-zero.

Key equations from Misra & Radhakrishna (1962):
- Field equations (46)-(49)
- General relation (eq 50): g₂₂ = -α - βφ + ½φ²
- Specific solutions (56)-(59)
"""

import sympy as sp
import numpy as np
from typing import Dict
from ..metric import EinsteinRosenMetric, rho, t

# Symbols for parameters
alpha_sym, beta_sym = sp.symbols('alpha beta', real=True)
m_sym, n_sym, a_sym = sp.symbols('m n a', real=True, positive=True)
p_sym, q_sym, b_sym = sp.symbols('p q b', real=True)


class CaseIISolution:
    """
    Case (ii) solution where ψ = 0 (only φ electromagnetic potential).

    Several specific sub-solutions are available from equations (56)-(59):
    - 'rho_only': φ depends only on ρ (eq 56)
    - 't_only': φ depends only on t (eq 59)
    - 'rho_quadratic': φ = ½aρ² + b (eq 57)
    - 't_linear': φ = at + b (eq 58)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.0,
                 variant: str = 'rho_only',
                 m: float = 1.0, n: float = 0.0, a: float = 1.0,
                 p: float = 0.0, q: float = 0.0, b: float = 0.0):
        """
        Initialize Case (ii) solution.

        Args:
            alpha: Parameter α in the general relation
            beta: Parameter β in the general relation
            variant: Solution variant ('rho_only', 't_only', 'rho_quadratic', 't_linear')
            m, n, a: Parameters for specific solutions
            p, q, b: Integration constants
        """
        self.alpha = alpha
        self.beta = beta
        self.variant = variant
        self.m = m
        self.n = n
        self.a = a
        self.p = p
        self.q = q
        self.b = b

        self._build_symbolic()
        self._create_numerical_functions()

    def _build_symbolic(self):
        """Build symbolic expressions for φ, μ, λ."""
        if self.variant == 'rho_only':
            self._build_rho_only_symbolic()
        elif self.variant == 't_only':
            self._build_t_only_symbolic()
        elif self.variant == 'rho_quadratic':
            self._build_rho_quadratic_symbolic()
        elif self.variant == 't_linear':
            self._build_t_linear_symbolic()
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def _build_rho_only_symbolic(self):
        """
        Solution where φ depends only on ρ (equation 56).

        φ = p - (m/a) * tanh(-(m/2)*log(ρ) + n)
        μ = log[(2a²/m²) * ρ² * cosh²(-(m/2)*log(ρ) + n)]
        λ = (m²/2)*log(ρ) + log[ρ² * cosh⁴(-(m/2)*log(ρ) + n)] + q
        """
        arg = -m_sym/2 * sp.log(rho) + n_sym

        self.phi_symbolic = p_sym - (m_sym/a_sym) * sp.tanh(arg)
        self.mu_symbolic = sp.log((2*a_sym**2 / m_sym**2) * rho**2 * sp.cosh(arg)**2)
        self.lambda_symbolic = (m_sym**2 / 2) * sp.log(rho) + sp.log(rho**2 * sp.cosh(arg)**4) + q_sym

    def _build_t_only_symbolic(self):
        """
        Solution where φ depends only on t (equation 59).

        φ = p - (m/a) * tanh(-(m/2)*t + n)
        μ = log[(2a²/m²) * ρ² * cosh²(-(m/2)*t + n)]
        λ = (m²ρ²)/4 + log[ρ² * cosh⁴(-(m/2)*t + n)] + q
        """
        arg = -m_sym/2 * t + n_sym

        self.phi_symbolic = p_sym - (m_sym/a_sym) * sp.tanh(arg)
        self.mu_symbolic = sp.log((2*a_sym**2 / m_sym**2) * rho**2 * sp.cosh(arg)**2)
        self.lambda_symbolic = (m_sym**2 * rho**2) / 4 + sp.log(rho**2 * sp.cosh(arg)**4) + q_sym

    def _build_rho_quadratic_symbolic(self):
        """
        Solution with φ quadratic in ρ (equation 57).

        φ = ½aρ² + b
        μ = log[(m²/2a²) * sech²(-(m/2)*t + n)]
        λ = (m²ρ²)/4 + p
        """
        arg = -m_sym/2 * t + n_sym
        sech_arg = 1 / sp.cosh(arg)

        self.phi_symbolic = sp.Rational(1, 2) * a_sym * rho**2 + b_sym
        self.mu_symbolic = sp.log((m_sym**2 / (2*a_sym**2)) * sech_arg**2)
        self.lambda_symbolic = (m_sym**2 * rho**2) / 4 + p_sym

    def _build_t_linear_symbolic(self):
        """
        Solution with φ linear in t (equation 58).

        φ = at + b
        μ = log[(m²/2a²) * sech²(-(m/2)*log(ρ) + n)]
        λ = (m²/2)*log(ρ) + p
        """
        arg = -m_sym/2 * sp.log(rho) + n_sym
        sech_arg = 1 / sp.cosh(arg)

        self.phi_symbolic = a_sym * t + b_sym
        self.mu_symbolic = sp.log((m_sym**2 / (2*a_sym**2)) * sech_arg**2)
        self.lambda_symbolic = (m_sym**2 / 2) * sp.log(rho) + p_sym

    def _create_numerical_functions(self):
        """Create fast numerical evaluation functions."""
        param_subs = {
            alpha_sym: self.alpha,
            beta_sym: self.beta,
            m_sym: self.m,
            n_sym: self.n,
            a_sym: self.a,
            p_sym: self.p,
            q_sym: self.q,
            b_sym: self.b,
        }

        phi_subst = self.phi_symbolic.subs(param_subs)
        mu_subst = self.mu_symbolic.subs(param_subs)
        lam_subst = self.lambda_symbolic.subs(param_subs)

        self._phi_func = sp.lambdify((rho, t), phi_subst, 'numpy')
        self._mu_func = sp.lambdify((rho, t), mu_subst, 'numpy')
        self._lambda_func = sp.lambdify((rho, t), lam_subst, 'numpy')

    def phi(self, rho_val: float, t_val: float) -> float:
        """Evaluate φ at (ρ, t)."""
        return float(self._phi_func(rho_val, t_val))

    def mu(self, rho_val: float, t_val: float) -> float:
        """Evaluate μ at (ρ, t)."""
        return float(self._mu_func(rho_val, t_val))

    def lambda_(self, rho_val: float, t_val: float) -> float:
        """Evaluate λ at (ρ, t)."""
        return float(self._lambda_func(rho_val, t_val))

    def psi(self, rho_val: float, t_val: float) -> float:
        """ψ is always 0 for Case (ii)."""
        return 0.0

    def get_metric(self) -> EinsteinRosenMetric:
        """Get the EinsteinRosenMetric object for this solution."""
        return EinsteinRosenMetric(self.lambda_symbolic, self.mu_symbolic)

    def metric_at(self, rho_val: float, t_val: float) -> np.ndarray:
        """Compute the metric tensor at a point."""
        lam = self.lambda_(rho_val, t_val)
        mu = self.mu(rho_val, t_val)

        exp_lam_minus_mu = np.exp(lam - mu)
        exp_minus_mu = np.exp(-mu)
        exp_mu = np.exp(mu)

        return np.array([
            [-exp_lam_minus_mu, 0, 0, 0],
            [0, -rho_val**2 * exp_minus_mu, 0, 0],
            [0, 0, -exp_mu, 0],
            [0, 0, 0, exp_lam_minus_mu]
        ])

    def field_tensor_at(self, rho_val: float, t_val: float) -> np.ndarray:
        """
        Compute the electromagnetic field tensor F_μν at a point.

        For Case (ii) with ψ = 0, from equation (9):
            F₁₂ = -φ₁/√(8π)
            F₁₃ = -ψ₁/√(8π) = 0
            F₂₄ = φ₄/√(8π)
            F₃₄ = ψ₄/√(8π) = 0
        """
        eps = 1e-6
        phi_1 = (self.phi(rho_val + eps, t_val) - self.phi(rho_val - eps, t_val)) / (2*eps)
        phi_4 = (self.phi(rho_val, t_val + eps) - self.phi(rho_val, t_val - eps)) / (2*eps)

        sqrt_8pi = np.sqrt(8 * np.pi)

        F = np.zeros((4, 4))
        # F₁₂ = -φ₁/√(8π) [indices: F[0,1]]
        F[0, 1] = -phi_1 / sqrt_8pi
        F[1, 0] = -F[0, 1]
        # F₂₄ = φ₄/√(8π) [indices: F[1,3]]
        F[1, 3] = phi_4 / sqrt_8pi
        F[3, 1] = -F[1, 3]

        return F

    def energy_tensor_at(self, rho_val: float, t_val: float) -> np.ndarray:
        """Compute electromagnetic energy tensor E^β_α at a point."""
        eps = 1e-6
        mu_val = self.mu(rho_val, t_val)
        lam_val = self.lambda_(rho_val, t_val)

        phi_1 = (self.phi(rho_val + eps, t_val) - self.phi(rho_val - eps, t_val)) / (2*eps)
        phi_4 = (self.phi(rho_val, t_val + eps) - self.phi(rho_val, t_val - eps)) / (2*eps)

        # From equation (10), for ψ = 0:
        exp_mu = np.exp(mu_val)
        factor = exp_mu / (16 * np.pi * rho_val**2)

        E44 = factor * (phi_1**2 + phi_4**2)
        E22 = factor * (phi_1**2 - phi_4**2)

        E = np.zeros((4, 4))
        E[0, 0] = -E44  # E¹₁
        E[1, 1] = E22   # E²₂
        E[2, 2] = -E22  # E³₃
        E[3, 3] = E44   # E⁴₄

        return E

    def verify_field_equations(self, rho_val: float, t_val: float) -> Dict[str, float]:
        """Check how well the solution satisfies the field equations."""
        eps = 1e-5
        r, tv = rho_val, t_val

        mu_val = self.mu(r, tv)

        # First derivatives
        mu_1 = (self.mu(r + eps, tv) - self.mu(r - eps, tv)) / (2*eps)
        mu_4 = (self.mu(r, tv + eps) - self.mu(r, tv - eps)) / (2*eps)
        phi_1 = (self.phi(r + eps, tv) - self.phi(r - eps, tv)) / (2*eps)
        phi_4 = (self.phi(r, tv + eps) - self.phi(r, tv - eps)) / (2*eps)

        # Second derivatives
        mu_11 = (self.mu(r + eps, tv) - 2*self.mu(r, tv) + self.mu(r - eps, tv)) / eps**2
        mu_44 = (self.mu(r, tv + eps) - 2*self.mu(r, tv) + self.mu(r, tv - eps)) / eps**2
        phi_11 = (self.phi(r + eps, tv) - 2*self.phi(r, tv) + self.phi(r - eps, tv)) / eps**2
        phi_44 = (self.phi(r, tv + eps) - 2*self.phi(r, tv) + self.phi(r, tv - eps)) / eps**2

        # Check equation (46): μ₁₁ - μ₄₄ + μ₁/ρ = (e^μ/ρ²)(φ₁² - φ₄²)
        lhs_46 = mu_11 - mu_44 + mu_1/r
        rhs_46 = (np.exp(mu_val) / r**2) * (phi_1**2 - phi_4**2)
        residual_46 = lhs_46 - rhs_46

        # Check equation (47): φ₁₁ - φ₄₄ - φ₁/ρ = -μ₁φ₁ + μ₄φ₄
        lhs_47 = phi_11 - phi_44 - phi_1/r
        rhs_47 = -mu_1*phi_1 + mu_4*phi_4
        residual_47 = lhs_47 - rhs_47

        return {
            'eq_46_residual': residual_46,
            'eq_47_residual': residual_47,
        }

    def __repr__(self) -> str:
        return (f"CaseIISolution(variant='{self.variant}', "
                f"m={self.m}, n={self.n}, a={self.a})")
