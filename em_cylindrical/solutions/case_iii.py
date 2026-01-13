"""
Case (iii): φ ≠ 0, ψ ≠ 0 Solutions

Both components of the electromagnetic 4-potential are non-zero.

Key equations from Misra & Radhakrishna (1962):
- Specific solutions (71)-(73), (78)

These solutions represent the most general electromagnetic fields
with both electric and magnetic components.
"""

import sympy as sp
import numpy as np
from typing import Dict
from ..metric import EinsteinRosenMetric, rho, t

# Symbols for parameters
m_sym, n_sym, a_sym = sp.symbols('m n a', real=True, positive=True)
p_sym, q_sym, b_sym = sp.symbols('p q b', real=True)


class CaseIIISolution:
    """
    Case (iii) solution where both φ and ψ are non-zero.

    Several specific sub-solutions are available from equations (71)-(73), (78):
    - 'variant_71': ψ = at + b, φ depends on ρ
    - 'variant_72': ψ depends on ρ, φ depends on t
    - 'variant_73': ψ depends on ρ, φ = at + b
    - 'variant_78': φ quadratic in ρ (related to Bonnor's theorem)
    """

    def __init__(self, variant: str = 'variant_71',
                 m: float = 1.0, n: float = 0.0, a: float = 1.0,
                 p: float = 0.0, q: float = 0.0, b: float = 0.0):
        """
        Initialize Case (iii) solution.

        Args:
            variant: Solution variant ('variant_71', 'variant_72', 'variant_73', 'variant_78')
            m, n, a: Parameters for specific solutions
            p, q, b: Integration constants
        """
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
        """Build symbolic expressions for φ, ψ, μ, λ."""
        if self.variant == 'variant_71':
            self._build_variant_71()
        elif self.variant == 'variant_72':
            self._build_variant_72()
        elif self.variant == 'variant_73':
            self._build_variant_73()
        elif self.variant == 'variant_78':
            self._build_variant_78()
        else:
            raise ValueError(f"Unknown variant: {self.variant}")

    def _build_variant_71(self):
        """
        Equation (71): ψ linear in t, φ depends on ρ.

        ψ = at + b
        φ = p - (m/2a) * tanh(-(m/2)*log(ρ) + n)
        μ = log[(4a²/m²) * ρ² * cosh²(-(m/2)*log(ρ) + n)]
        λ = (m²/2)*log(ρ) + log[ρ² * cosh⁴(-(m/2)*log(ρ) + n)] + q
        """
        arg = -m_sym/2 * sp.log(rho) + n_sym

        self.psi_symbolic = a_sym * t + b_sym
        self.phi_symbolic = p_sym - (m_sym / (2*a_sym)) * sp.tanh(arg)
        self.mu_symbolic = sp.log((4*a_sym**2 / m_sym**2) * rho**2 * sp.cosh(arg)**2)
        self.lambda_symbolic = (m_sym**2 / 2) * sp.log(rho) + sp.log(rho**2 * sp.cosh(arg)**4) + q_sym

    def _build_variant_72(self):
        """
        Equation (72): ψ quadratic in ρ, φ depends on t.

        ψ = ½aρ² + b
        φ = p - (m/2a) * tanh(-(m/2)*t + n)
        μ = log[(4a²/m²) * ρ² * cosh²(-(m/2)*t + n)]
        λ = (m²ρ²)/4 + log[ρ² * cosh⁴(-(m/2)*t + n)] + q
        """
        arg = -m_sym/2 * t + n_sym

        self.psi_symbolic = sp.Rational(1, 2) * a_sym * rho**2 + b_sym
        self.phi_symbolic = p_sym - (m_sym / (2*a_sym)) * sp.tanh(arg)
        self.mu_symbolic = sp.log((4*a_sym**2 / m_sym**2) * rho**2 * sp.cosh(arg)**2)
        self.lambda_symbolic = (m_sym**2 * rho**2) / 4 + sp.log(rho**2 * sp.cosh(arg)**4) + q_sym

    def _build_variant_73(self):
        """
        Equation (73): ψ depends on ρ (tanh form), φ linear in t.

        ψ = p - (m/2a) * tanh(-(m/2)*log(ρ) + n)
        φ = at + b
        μ = log[(m²/4a²) * sech²(-(m/2)*log(ρ) + n)]
        λ = (m²/2)*log(ρ) + q
        """
        arg = -m_sym/2 * sp.log(rho) + n_sym
        sech_arg = 1 / sp.cosh(arg)

        self.psi_symbolic = p_sym - (m_sym / (2*a_sym)) * sp.tanh(arg)
        self.phi_symbolic = a_sym * t + b_sym
        self.mu_symbolic = sp.log((m_sym**2 / (4*a_sym**2)) * sech_arg**2)
        self.lambda_symbolic = (m_sym**2 / 2) * sp.log(rho) + q_sym

    def _build_variant_78(self):
        """
        Equation (78): φ quadratic in ρ (Bonnor's theorem application).

        φ = ½aρ² + b
        μ = log[(m²/2a²) * sech²(-(m/2)*t + n)]
        λ = (m²ρ²)/4 + q

        Note: ψ is related through Bonnor's theorem; here we use ψ = 0
        as this is derivable from Case (ii B) via the theorem.
        """
        arg = -m_sym/2 * t + n_sym
        sech_arg = 1 / sp.cosh(arg)

        self.phi_symbolic = sp.Rational(1, 2) * a_sym * rho**2 + b_sym
        self.psi_symbolic = sp.Integer(0)  # Derived from case (ii B)
        self.mu_symbolic = sp.log((m_sym**2 / (2*a_sym**2)) * sech_arg**2)
        self.lambda_symbolic = (m_sym**2 * rho**2) / 4 + q_sym

    def _create_numerical_functions(self):
        """Create fast numerical evaluation functions."""
        param_subs = {
            m_sym: self.m,
            n_sym: self.n,
            a_sym: self.a,
            p_sym: self.p,
            q_sym: self.q,
            b_sym: self.b,
        }

        phi_subst = self.phi_symbolic.subs(param_subs)
        psi_subst = self.psi_symbolic.subs(param_subs)
        mu_subst = self.mu_symbolic.subs(param_subs)
        lam_subst = self.lambda_symbolic.subs(param_subs)

        self._phi_func = sp.lambdify((rho, t), phi_subst, 'numpy')
        self._psi_func = sp.lambdify((rho, t), psi_subst, 'numpy')
        self._mu_func = sp.lambdify((rho, t), mu_subst, 'numpy')
        self._lambda_func = sp.lambdify((rho, t), lam_subst, 'numpy')

    def phi(self, rho_val: float, t_val: float) -> float:
        """Evaluate φ at (ρ, t)."""
        return float(self._phi_func(rho_val, t_val))

    def psi(self, rho_val: float, t_val: float) -> float:
        """Evaluate ψ at (ρ, t)."""
        return float(self._psi_func(rho_val, t_val))

    def mu(self, rho_val: float, t_val: float) -> float:
        """Evaluate μ at (ρ, t)."""
        return float(self._mu_func(rho_val, t_val))

    def lambda_(self, rho_val: float, t_val: float) -> float:
        """Evaluate λ at (ρ, t)."""
        return float(self._lambda_func(rho_val, t_val))

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

        From equation (9), all four non-vanishing components:
            F₁₂ = -φ₁/√(8π)
            F₁₃ = -ψ₁/√(8π)
            F₂₄ = φ₄/√(8π)
            F₃₄ = ψ₄/√(8π)
        """
        eps = 1e-6

        # Derivatives of φ
        phi_1 = (self.phi(rho_val + eps, t_val) - self.phi(rho_val - eps, t_val)) / (2*eps)
        phi_4 = (self.phi(rho_val, t_val + eps) - self.phi(rho_val, t_val - eps)) / (2*eps)

        # Derivatives of ψ
        psi_1 = (self.psi(rho_val + eps, t_val) - self.psi(rho_val - eps, t_val)) / (2*eps)
        psi_4 = (self.psi(rho_val, t_val + eps) - self.psi(rho_val, t_val - eps)) / (2*eps)

        sqrt_8pi = np.sqrt(8 * np.pi)

        F = np.zeros((4, 4))
        # F₁₂ = -φ₁/√(8π) [indices: F[0,1]]
        F[0, 1] = -phi_1 / sqrt_8pi
        F[1, 0] = -F[0, 1]
        # F₁₃ = -ψ₁/√(8π) [indices: F[0,2]]
        F[0, 2] = -psi_1 / sqrt_8pi
        F[2, 0] = -F[0, 2]
        # F₂₄ = φ₄/√(8π) [indices: F[1,3]]
        F[1, 3] = phi_4 / sqrt_8pi
        F[3, 1] = -F[1, 3]
        # F₃₄ = ψ₄/√(8π) [indices: F[2,3]]
        F[2, 3] = psi_4 / sqrt_8pi
        F[3, 2] = -F[2, 3]

        return F

    def energy_tensor_at(self, rho_val: float, t_val: float) -> np.ndarray:
        """
        Compute electromagnetic energy tensor E^β_α at a point.

        From equations (10), for both φ and ψ non-zero.
        """
        eps = 1e-6
        mu_val = self.mu(rho_val, t_val)
        lam_val = self.lambda_(rho_val, t_val)

        # Derivatives
        phi_1 = (self.phi(rho_val + eps, t_val) - self.phi(rho_val - eps, t_val)) / (2*eps)
        phi_4 = (self.phi(rho_val, t_val + eps) - self.phi(rho_val, t_val - eps)) / (2*eps)
        psi_1 = (self.psi(rho_val + eps, t_val) - self.psi(rho_val - eps, t_val)) / (2*eps)
        psi_4 = (self.psi(rho_val, t_val + eps) - self.psi(rho_val, t_val - eps)) / (2*eps)

        exp_mu = np.exp(mu_val)
        exp_lam = np.exp(lam_val)
        r = rho_val

        # From equation (10):
        # E⁴₄ = -E¹₁ = (1/16π)e^(-λ)[(1/ρ²)e^(2μ)(φ₁² + φ₄²) + ψ₁² + ψ₄²]
        # E²₂ = -E³₃ = (1/16π)e^(-λ)[-(1/ρ²)e^(2μ)(φ₁² - φ₄²) + ψ₁² - ψ₄²]

        exp_minus_lam = np.exp(-lam_val)
        exp_2mu = np.exp(2*mu_val)

        E44 = (1/(16*np.pi)) * exp_minus_lam * (
            (1/r**2) * exp_2mu * (phi_1**2 + phi_4**2) + psi_1**2 + psi_4**2
        )
        E22 = (1/(16*np.pi)) * exp_minus_lam * (
            -(1/r**2) * exp_2mu * (phi_1**2 - phi_4**2) + psi_1**2 - psi_4**2
        )

        E = np.zeros((4, 4))
        E[0, 0] = -E44  # E¹₁
        E[1, 1] = E22   # E²₂
        E[2, 2] = -E22  # E³₃
        E[3, 3] = E44   # E⁴₄

        return E

    def verify_field_equations(self, rho_val: float, t_val: float) -> Dict[str, float]:
        """
        Check how well the solution satisfies the field equations (11)-(17).

        Returns dictionary of residuals.
        """
        eps = 1e-5
        r, tv = rho_val, t_val

        mu_val = self.mu(r, tv)

        # First derivatives
        mu_1 = (self.mu(r + eps, tv) - self.mu(r - eps, tv)) / (2*eps)
        mu_4 = (self.mu(r, tv + eps) - self.mu(r, tv - eps)) / (2*eps)
        phi_1 = (self.phi(r + eps, tv) - self.phi(r - eps, tv)) / (2*eps)
        phi_4 = (self.phi(r, tv + eps) - self.phi(r, tv - eps)) / (2*eps)
        psi_1 = (self.psi(r + eps, tv) - self.psi(r - eps, tv)) / (2*eps)
        psi_4 = (self.psi(r, tv + eps) - self.psi(r, tv - eps)) / (2*eps)

        # Second derivatives
        mu_11 = (self.mu(r + eps, tv) - 2*self.mu(r, tv) + self.mu(r - eps, tv)) / eps**2
        mu_44 = (self.mu(r, tv + eps) - 2*self.mu(r, tv) + self.mu(r, tv - eps)) / eps**2
        phi_11 = (self.phi(r + eps, tv) - 2*self.phi(r, tv) + self.phi(r - eps, tv)) / eps**2
        phi_44 = (self.phi(r, tv + eps) - 2*self.phi(r, tv) + self.phi(r, tv - eps)) / eps**2
        psi_11 = (self.psi(r + eps, tv) - 2*self.psi(r, tv) + self.psi(r - eps, tv)) / eps**2
        psi_44 = (self.psi(r, tv + eps) - 2*self.psi(r, tv) + self.psi(r, tv - eps)) / eps**2

        exp_mu = np.exp(mu_val)

        # Check equation (12): μ₁₁ - μ₄₄ + μ₁/ρ = (e^μ/ρ²)(φ₁² - φ₄²) - e^(-μ)(ψ₁² - ψ₄²)
        lhs_12 = mu_11 - mu_44 + mu_1/r
        rhs_12 = (exp_mu/r**2)*(phi_1**2 - phi_4**2) - np.exp(-mu_val)*(psi_1**2 - psi_4**2)
        residual_12 = lhs_12 - rhs_12

        # Check equation (15): φ₁ψ₁ - φ₄ψ₄ = 0
        residual_15 = phi_1*psi_1 - phi_4*psi_4

        # Check equations (16) and (17)
        lhs_16 = phi_11 - phi_44 - phi_1/r
        rhs_16 = -mu_1*phi_1 + mu_4*phi_4
        residual_16 = lhs_16 - rhs_16

        lhs_17 = psi_11 - psi_44 + psi_1/r
        rhs_17 = mu_1*psi_1 - mu_4*psi_4
        residual_17 = lhs_17 - rhs_17

        return {
            'eq_12_residual': residual_12,
            'eq_15_residual': residual_15,
            'eq_16_residual': residual_16,
            'eq_17_residual': residual_17,
        }

    def __repr__(self) -> str:
        return (f"CaseIIISolution(variant='{self.variant}', "
                f"m={self.m}, n={self.n}, a={self.a})")
