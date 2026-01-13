"""
Case (i): φ = 0 Solutions

When φ = 0, only the ψ component of the electromagnetic 4-potential is non-zero.

Key equations from Misra & Radhakrishna (1962):
- General relation (eq 22): g₃₃ = -α - βψ + ½ψ²
- Wave equation for x (eq 29): x₁₁ - x₄₄ + x₁/ρ = 0
- Solution (eqs 30-33):
    ψ = β + √(2α+β²) tanh(x)
    e^μ = (α + ½β²) sech²(x)
    λ₁ = 2ρ(x₁² + x₄²)
    λ₄ = 4ρx₁x₄
"""

import sympy as sp
import numpy as np
from typing import Optional, Dict, Any
from scipy.special import j0, j1  # Bessel functions
from ..metric import EinsteinRosenMetric, rho, t

# Additional symbols for this case
alpha_sym, beta_sym = sp.symbols('alpha beta', real=True)
m_sym, n_sym, a_sym = sp.symbols('m n a', real=True, positive=True)
p_sym, q_sym = sp.symbols('p q', real=True)
k_sym, A_sym, epsilon_sym = sp.symbols('k A epsilon', real=True)


class CaseISolution:
    """
    Case (i) solution where φ = 0 (only ψ electromagnetic potential).

    This implements the general solution given by equations (30)-(33) where
    the variable x satisfies the cylindrical wave equation.

    Several specific sub-solutions are available:
    - 'bessel': x = A*J₀(kρ)*cos(kt + ε) - Bessel function solution
    - 't_only': ψ depends only on t (eq 43)
    - 'rho_only': ψ depends only on ρ (eq 44 variant)
    """

    def __init__(self, alpha: float, beta: float,
                 variant: str = 'bessel',
                 m: float = 1.0, n: float = 0.0, a: float = 1.0,
                 p: float = 0.0, q: float = 0.0,
                 k: float = 1.0, A: float = 1.0, epsilon: float = 0.0):
        """
        Initialize Case (i) solution.

        Args:
            alpha: Parameter α in the general relation (must satisfy α > 0)
            beta: Parameter β in the general relation
            variant: Solution variant ('bessel', 't_only', 'rho_only')
            m, n, a: Parameters for specific solutions
            p, q: Integration constants for λ
            k, A, epsilon: Bessel solution parameters (x = A*J₀(kρ)*cos(kt+ε))
        """
        self.alpha = alpha
        self.beta = beta
        self.variant = variant
        self.m = m
        self.n = n
        self.a = a
        self.p = p
        self.q = q
        self.k = k
        self.A = A
        self.epsilon = epsilon

        # Derived quantity
        self._sqrt_term = np.sqrt(2*alpha + beta**2)

        # Build symbolic expressions
        self._build_symbolic()

        # Create numerical functions
        self._create_numerical_functions()

    def _build_symbolic(self):
        """Build symbolic expressions for ψ, μ, λ."""
        if self.variant == 'bessel':
            self._build_bessel_symbolic()
        elif self.variant == 't_only':
            self._build_t_only_symbolic()
        elif self.variant == 'rho_only':
            self._build_rho_only_symbolic()
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def _build_bessel_symbolic(self):
        """
        Bessel function solution: x = A*J₀(kρ)*cos(kt + ε)

        From equations (30)-(33):
            ψ = β + √(2α+β²) tanh(x)
            e^μ = (α + ½β²) sech²(x)
        """
        # x = A * J₀(kρ) * cos(kt + ε)
        # We'll use symbolic Bessel function
        x = A_sym * sp.besselj(0, k_sym * rho) * sp.cos(k_sym * t + epsilon_sym)

        # ψ = β + √(2α + β²) * tanh(x)  [eq 30]
        sqrt_term = sp.sqrt(2*alpha_sym + beta_sym**2)
        self.psi_symbolic = beta_sym + sqrt_term * sp.tanh(x)

        # e^μ = (α + ½β²) * sech²(x)  [eq 33]
        sech_x = 1 / sp.cosh(x)
        exp_mu = (alpha_sym + sp.Rational(1, 2)*beta_sym**2) * sech_x**2
        self.mu_symbolic = sp.log(exp_mu)

        # For λ, we need to integrate (31) and (32)
        # λ₁ = 2ρ(x₁² + x₄²), λ₄ = 4ρx₁x₄
        # This is complex for Bessel functions, so we compute numerically
        x1 = sp.diff(x, rho)
        x4 = sp.diff(x, t)
        self.lambda_1_symbolic = 2 * rho * (x1**2 + x4**2)
        self.lambda_4_symbolic = 4 * rho * x1 * x4

        # For now, λ requires numerical integration
        # We'll use an approximate form
        self.lambda_symbolic = q_sym  # Placeholder - computed numerically

        self._x_symbolic = x

    def _build_t_only_symbolic(self):
        """
        Solution where ψ depends only on t (equation 43).

        ψ = p - (m/a) * tanh(-(m/2)*t + n)
        μ = log[(m²/2a²) * sech²(-(m/2)*t + n)]
        λ = (m²ρ²)/4 + q
        """
        arg = -m_sym/2 * t + n_sym

        # ψ from eq 43
        self.psi_symbolic = p_sym - (m_sym/a_sym) * sp.tanh(arg)

        # μ from eq 43
        sech_arg = 1 / sp.cosh(arg)
        self.mu_symbolic = sp.log((m_sym**2 / (2*a_sym**2)) * sech_arg**2)

        # λ from eq 43
        self.lambda_symbolic = (m_sym**2 * rho**2) / 4 + q_sym

        self._x_symbolic = None

    def _build_rho_only_symbolic(self):
        """
        Solution where ψ depends only on ρ (equation 44 variant).

        ψ = a*t + b  (linear in t)
        μ = log[(2a²/m²) * ρ² * cosh²((m/2)*log(ρ) + n)]
        λ = (m²/2)*log(ρ) + log[ρ² * cosh⁴((m/2)*log(ρ) + n)] + p
        """
        # Note: In this variant, ψ is actually linear in t, and φ depends on ρ
        # This follows equation (44) in the paper
        b_sym = sp.Symbol('b', real=True)

        arg = m_sym/2 * sp.log(rho) + n_sym

        # ψ = at + b (linear in t)
        self.psi_symbolic = a_sym * t + b_sym

        # μ from eq 44
        self.mu_symbolic = sp.log((2*a_sym**2 / m_sym**2) * rho**2 * sp.cosh(arg)**2)

        # λ from eq 44
        self.lambda_symbolic = (m_sym**2 / 2) * sp.log(rho) + sp.log(rho**2 * sp.cosh(arg)**4) + p_sym

        self._x_symbolic = None

    def _create_numerical_functions(self):
        """Create fast numerical evaluation functions."""
        # Substitution dictionary for parameters
        param_subs = {
            alpha_sym: self.alpha,
            beta_sym: self.beta,
            m_sym: self.m,
            n_sym: self.n,
            a_sym: self.a,
            p_sym: self.p,
            q_sym: self.q,
            k_sym: self.k,
            A_sym: self.A,
            epsilon_sym: self.epsilon,
        }

        if self.variant == 'bessel':
            # For Bessel variant, use custom numerical implementation
            self._psi_func = self._psi_bessel_numerical
            self._mu_func = self._mu_bessel_numerical
            self._lambda_func = self._lambda_bessel_numerical
        else:
            # Substitute parameters and lambdify
            psi_subst = self.psi_symbolic.subs(param_subs)
            mu_subst = self.mu_symbolic.subs(param_subs)
            lam_subst = self.lambda_symbolic.subs(param_subs)

            # Handle additional symbols like 'b' for rho_only variant
            if self.variant == 'rho_only':
                b_sym = sp.Symbol('b', real=True)
                psi_subst = psi_subst.subs(b_sym, self.p)  # Use p as b

            self._psi_func = sp.lambdify((rho, t), psi_subst, 'numpy')
            self._mu_func = sp.lambdify((rho, t), mu_subst, 'numpy')
            self._lambda_func = sp.lambdify((rho, t), lam_subst, 'numpy')

    def _psi_bessel_numerical(self, rho_val: float, t_val: float) -> float:
        """Numerical ψ for Bessel variant."""
        x = self.A * j0(self.k * rho_val) * np.cos(self.k * t_val + self.epsilon)
        return self.beta + self._sqrt_term * np.tanh(x)

    def _mu_bessel_numerical(self, rho_val: float, t_val: float) -> float:
        """Numerical μ for Bessel variant."""
        x = self.A * j0(self.k * rho_val) * np.cos(self.k * t_val + self.epsilon)
        sech_x = 1 / np.cosh(x)
        exp_mu = (self.alpha + 0.5 * self.beta**2) * sech_x**2
        return np.log(exp_mu)

    def _lambda_bessel_numerical(self, rho_val: float, t_val: float) -> float:
        """
        Numerical λ for Bessel variant.

        λ is obtained by integrating:
            λ₁ = 2ρ(x₁² + x₄²)
            λ₄ = 4ρx₁x₄

        For the Bessel solution x = A*J₀(kρ)*cos(kt+ε):
            x₁ = -Ak*J₁(kρ)*cos(kt+ε)  (derivative of J₀ is -J₁)
            x₄ = -Ak*J₀(kρ)*sin(kt+ε)
        """
        cos_term = np.cos(self.k * t_val + self.epsilon)
        sin_term = np.sin(self.k * t_val + self.epsilon)

        # Derivatives of x
        x1 = -self.A * self.k * j1(self.k * rho_val) * cos_term
        x4 = -self.A * self.k * j0(self.k * rho_val) * sin_term

        # λ₁ and λ₄
        lambda_1 = 2 * rho_val * (x1**2 + x4**2)

        # For simplicity, return an approximate λ based on λ₁ at the point
        # A more complete implementation would integrate the PDEs
        return rho_val * lambda_1 / 2 + self.q

    def psi(self, rho_val: float, t_val: float) -> float:
        """Evaluate ψ at (ρ, t)."""
        return float(self._psi_func(rho_val, t_val))

    def mu(self, rho_val: float, t_val: float) -> float:
        """Evaluate μ at (ρ, t)."""
        return float(self._mu_func(rho_val, t_val))

    def lambda_(self, rho_val: float, t_val: float) -> float:
        """Evaluate λ at (ρ, t)."""
        return float(self._lambda_func(rho_val, t_val))

    def phi(self, rho_val: float, t_val: float) -> float:
        """φ is always 0 for Case (i)."""
        return 0.0

    def get_metric(self) -> EinsteinRosenMetric:
        """Get the EinsteinRosenMetric object for this solution."""
        return EinsteinRosenMetric(self.lambda_symbolic, self.mu_symbolic)

    def metric_at(self, rho_val: float, t_val: float) -> np.ndarray:
        """
        Compute the metric tensor at a point.

        Returns:
            4x4 numpy array of metric components g_ij
        """
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

        For Case (i) with φ = 0, from equation (9):
            F₁₂ = -φ₁/√(8π) = 0
            F₁₃ = -ψ₁/√(8π)
            F₂₄ = φ₄/√(8π) = 0
            F₃₄ = ψ₄/√(8π)

        Returns:
            4x4 antisymmetric numpy array
        """
        # Numerical derivatives
        eps = 1e-6
        psi_1 = (self.psi(rho_val + eps, t_val) - self.psi(rho_val - eps, t_val)) / (2*eps)
        psi_4 = (self.psi(rho_val, t_val + eps) - self.psi(rho_val, t_val - eps)) / (2*eps)

        sqrt_8pi = np.sqrt(8 * np.pi)

        F = np.zeros((4, 4))
        # F₁₃ = -ψ₁/√(8π) [indices: F[0,2]]
        F[0, 2] = -psi_1 / sqrt_8pi
        F[2, 0] = -F[0, 2]
        # F₃₄ = ψ₄/√(8π) [indices: F[2,3]]
        F[2, 3] = psi_4 / sqrt_8pi
        F[3, 2] = -F[2, 3]

        return F

    def energy_tensor_at(self, rho_val: float, t_val: float) -> np.ndarray:
        """
        Compute electromagnetic energy tensor E^β_α at a point.

        From equations (10) in the paper.
        """
        eps = 1e-6
        mu_val = self.mu(rho_val, t_val)
        lam_val = self.lambda_(rho_val, t_val)

        # Derivatives of ψ
        psi_1 = (self.psi(rho_val + eps, t_val) - self.psi(rho_val - eps, t_val)) / (2*eps)
        psi_4 = (self.psi(rho_val, t_val + eps) - self.psi(rho_val, t_val - eps)) / (2*eps)

        exp_lam = np.exp(lam_val)
        exp_mu = np.exp(mu_val)

        # From equation (10), for φ = 0:
        # E⁴₄ = -E¹₁ = (1/16π) * e^(-λ) * [ψ₁² + ψ₄²]  (simplified)
        # E²₂ = -E³₃ = (1/16π) * e^(-λ) * [-ψ₁² + ψ₄²] (simplified)

        factor = 1 / (16 * np.pi) * np.exp(-lam_val)
        E44 = factor * (psi_1**2 + psi_4**2)
        E22 = factor * (-psi_1**2 + psi_4**2)

        E = np.zeros((4, 4))
        E[0, 0] = -E44  # E¹₁
        E[1, 1] = E22   # E²₂
        E[2, 2] = -E22  # E³₃
        E[3, 3] = E44   # E⁴₄

        return E

    def verify_field_equations(self, rho_val: float, t_val: float) -> Dict[str, float]:
        """
        Check how well the solution satisfies the field equations.

        Returns dictionary of residuals (should be close to 0 for valid solutions).
        """
        eps = 1e-5
        r, tv = rho_val, t_val

        # Get function values
        mu_val = self.mu(r, tv)
        psi_val = self.psi(r, tv)

        # First derivatives
        mu_1 = (self.mu(r + eps, tv) - self.mu(r - eps, tv)) / (2*eps)
        mu_4 = (self.mu(r, tv + eps) - self.mu(r, tv - eps)) / (2*eps)
        psi_1 = (self.psi(r + eps, tv) - self.psi(r - eps, tv)) / (2*eps)
        psi_4 = (self.psi(r, tv + eps) - self.psi(r, tv - eps)) / (2*eps)

        # Second derivatives
        mu_11 = (self.mu(r + eps, tv) - 2*self.mu(r, tv) + self.mu(r - eps, tv)) / eps**2
        mu_44 = (self.mu(r, tv + eps) - 2*self.mu(r, tv) + self.mu(r, tv - eps)) / eps**2
        psi_11 = (self.psi(r + eps, tv) - 2*self.psi(r, tv) + self.psi(r - eps, tv)) / eps**2
        psi_44 = (self.psi(r, tv + eps) - 2*self.psi(r, tv) + self.psi(r, tv - eps)) / eps**2

        # Check equation (18): μ₁₁ - μ₄₄ + μ₁/ρ = -e^(-μ)(ψ₁² - ψ₄²)
        lhs_18 = mu_11 - mu_44 + mu_1/r
        rhs_18 = -np.exp(-mu_val) * (psi_1**2 - psi_4**2)
        residual_18 = lhs_18 - rhs_18

        # Check equation (19): ψ₁₁ - ψ₄₄ + ψ₁/ρ = μ₁ψ₁ - μ₄ψ₄
        lhs_19 = psi_11 - psi_44 + psi_1/r
        rhs_19 = mu_1*psi_1 - mu_4*psi_4
        residual_19 = lhs_19 - rhs_19

        return {
            'eq_18_residual': residual_18,
            'eq_19_residual': residual_19,
        }

    def __repr__(self) -> str:
        return (f"CaseISolution(alpha={self.alpha}, beta={self.beta}, "
                f"variant='{self.variant}')")
