"""
Electromagnetic Energy-Momentum Tensor E^β_α

From equation (4) in Misra & Radhakrishna (1962):
    E^β_α = -F_μα F^μβ + ¼δ^β_α F_σε F^σε

The non-vanishing independent components are given by equation (10).
"""

import sympy as sp
import numpy as np
from typing import Dict

# Coordinate symbols
rho, t = sp.symbols('rho t', real=True, positive=True)


class ElectromagneticEnergyTensor:
    """
    Electromagnetic energy-momentum tensor E^β_α.

    This is the source term in the Einstein field equations (2):
        R^β_α = -8π E^β_α

    Coordinates: (x¹, x², x³, x⁴) = (ρ, Φ, z, t)
    """

    def __init__(self, phi_expr: sp.Expr, psi_expr: sp.Expr,
                 mu_expr: sp.Expr, lambda_expr: sp.Expr):
        """
        Initialize energy tensor from solution components.

        Args:
            phi_expr: SymPy expression for φ(ρ, t)
            psi_expr: SymPy expression for ψ(ρ, t)
            mu_expr: SymPy expression for μ(ρ, t)
            lambda_expr: SymPy expression for λ(ρ, t)
        """
        self.phi_symbolic = phi_expr
        self.psi_symbolic = psi_expr
        self.mu_symbolic = mu_expr
        self.lambda_symbolic = lambda_expr

        self._build_tensor()
        self._create_numerical_functions()

    def _build_tensor(self):
        """
        Build symbolic energy tensor components from equation (10).

        E⁴₄ = -E¹₁ = (1/16π)e^(-λ)[(1/ρ²)e^(2μ)(φ₁² + φ₄²) + ψ₁² + ψ₄²]
        E²₂ = -E³₃ = (1/16π)e^(-λ)[-(1/ρ²)e^(2μ)(φ₁² - φ₄²) + ψ₁² - ψ₄²]
        E¹₄ = E⁴₁ = (1/8π)e^(-λ)[(1/ρ²)e^(2μ)φ₁φ₄ + ψ₁ψ₄]
        E²₃ = (1/8π)e^(-λ)[-φ₁ψ₁ + φ₄ψ₄]
        """
        # Derivatives
        phi_1 = sp.diff(self.phi_symbolic, rho)
        phi_4 = sp.diff(self.phi_symbolic, t)
        psi_1 = sp.diff(self.psi_symbolic, rho)
        psi_4 = sp.diff(self.psi_symbolic, t)

        mu = self.mu_symbolic
        lam = self.lambda_symbolic

        exp_minus_lam = sp.exp(-lam)
        exp_2mu = sp.exp(2*mu)

        # Component expressions from eq (10)
        # E⁴₄ = -E¹₁
        self.E44_symbolic = (1/(16*sp.pi)) * exp_minus_lam * (
            (1/rho**2) * exp_2mu * (phi_1**2 + phi_4**2) + psi_1**2 + psi_4**2
        )
        self.E11_symbolic = -self.E44_symbolic

        # E²₂ = -E³₃
        self.E22_symbolic = (1/(16*sp.pi)) * exp_minus_lam * (
            -(1/rho**2) * exp_2mu * (phi_1**2 - phi_4**2) + psi_1**2 - psi_4**2
        )
        self.E33_symbolic = -self.E22_symbolic

        # E¹₄ = E⁴₁
        self.E14_symbolic = (1/(8*sp.pi)) * exp_minus_lam * (
            (1/rho**2) * exp_2mu * phi_1 * phi_4 + psi_1 * psi_4
        )

        # E²₃
        self.E23_symbolic = (1/(8*sp.pi)) * exp_minus_lam * (
            -phi_1 * psi_1 + phi_4 * psi_4
        )

    def _create_numerical_functions(self):
        """Create numerical evaluation functions."""
        self._E11_func = sp.lambdify((rho, t), self.E11_symbolic, 'numpy')
        self._E22_func = sp.lambdify((rho, t), self.E22_symbolic, 'numpy')
        self._E33_func = sp.lambdify((rho, t), self.E33_symbolic, 'numpy')
        self._E44_func = sp.lambdify((rho, t), self.E44_symbolic, 'numpy')
        self._E14_func = sp.lambdify((rho, t), self.E14_symbolic, 'numpy')
        self._E23_func = sp.lambdify((rho, t), self.E23_symbolic, 'numpy')

    def at(self, rho_val: float, t_val: float) -> np.ndarray:
        """
        Evaluate the energy tensor E^β_α numerically at a point.

        Returns:
            4x4 numpy array of mixed tensor components
        """
        E = np.zeros((4, 4))
        E[0, 0] = self._E11_func(rho_val, t_val)  # E¹₁
        E[1, 1] = self._E22_func(rho_val, t_val)  # E²₂
        E[2, 2] = self._E33_func(rho_val, t_val)  # E³₃
        E[3, 3] = self._E44_func(rho_val, t_val)  # E⁴₄
        E[0, 3] = self._E14_func(rho_val, t_val)  # E¹₄
        E[3, 0] = E[0, 3]                          # E⁴₁ = E¹₄
        E[1, 2] = self._E23_func(rho_val, t_val)  # E²₃

        return E

    def trace(self, rho_val: float, t_val: float) -> float:
        """
        Compute the trace E^α_α.

        For electromagnetic fields, the trace should vanish.
        """
        E = self.at(rho_val, t_val)
        return np.trace(E)

    def energy_conditions(self, rho_val: float, t_val: float) -> Dict[str, bool]:
        """
        Check various energy conditions at a point.

        Returns dictionary indicating which conditions are satisfied.
        """
        E = self.at(rho_val, t_val)

        # Weak energy condition: ρ ≥ 0 (energy density non-negative)
        # For EM fields, this is related to E⁴₄
        rho_energy = E[3, 3]

        # Dominant energy condition: energy flux should be timelike or null
        # Strong energy condition: ρ + 3p ≥ 0 and ρ + p ≥ 0

        return {
            'weak_energy': rho_energy >= 0,
            'trace_free': abs(np.trace(E)) < 1e-10,
        }


def compute_energy_tensor_from_solution(solution) -> ElectromagneticEnergyTensor:
    """
    Create an energy tensor from any solution object.

    Args:
        solution: A CaseISolution, CaseIISolution, or CaseIIISolution object

    Returns:
        ElectromagneticEnergyTensor instance
    """
    return ElectromagneticEnergyTensor(
        solution.phi_symbolic,
        solution.psi_symbolic,
        solution.mu_symbolic,
        solution.lambda_symbolic
    )
