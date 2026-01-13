"""
Wave Equation Solutions for Cylindrical Symmetry

The wave equation (29) from Misra & Radhakrishna (1962):
    x₁₁ - x₄₄ + x₁/ρ = 0

where subscripts 1 and 4 denote partial differentiation with respect
to ρ and t respectively.

This is the cylindrical wave equation in (2+1) dimensions, whose
general solution can be expressed in terms of Bessel functions.
"""

import sympy as sp
import numpy as np
from scipy.special import j0, j1, y0, y1, jn, yn
from typing import Callable, List, Tuple

# Symbols
rho, t = sp.symbols('rho t', real=True, positive=True)
k_sym, A_sym, B_sym, epsilon_sym = sp.symbols('k A B epsilon', real=True)


class CylindricalWaveSolution:
    """
    Solutions to the cylindrical wave equation.

    The general solution is a superposition of modes:
        x = sum_n [A_n * J_0(k_n * rho) + B_n * Y_0(k_n * rho)] * cos(k_n * t + epsilon_n)

    where J_0 and Y_0 are Bessel functions of the first and second kind.
    """

    def __init__(self, modes: List[Tuple[float, float, float, float]] = None):
        """
        Initialize with a list of wave modes.

        Args:
            modes: List of (k, A, B, epsilon) tuples for each mode.
                   k = wave number
                   A = coefficient of J_0(k*rho)
                   B = coefficient of Y_0(k*rho) (usually 0 for regularity at rho=0)
                   epsilon = phase
        """
        if modes is None:
            # Default single mode
            modes = [(1.0, 1.0, 0.0, 0.0)]

        self.modes = modes
        self._build_symbolic()

    def _build_symbolic(self):
        """Build symbolic expression for x."""
        x_expr = sp.Integer(0)
        for k, A, B, eps in self.modes:
            k_val = sp.Float(k)
            A_val = sp.Float(A)
            B_val = sp.Float(B)
            eps_val = sp.Float(eps)

            mode = (A_val * sp.besselj(0, k_val * rho) +
                    B_val * sp.bessely(0, k_val * rho)) * sp.cos(k_val * t + eps_val)
            x_expr += mode

        self.x_symbolic = x_expr

        # Derivatives
        self.x_1_symbolic = sp.diff(x_expr, rho)
        self.x_4_symbolic = sp.diff(x_expr, t)
        self.x_11_symbolic = sp.diff(x_expr, rho, 2)
        self.x_44_symbolic = sp.diff(x_expr, t, 2)

    def x(self, rho_val: float, t_val: float) -> float:
        """Evaluate x at (rho, t)."""
        result = 0.0
        for k, A, B, eps in self.modes:
            cos_term = np.cos(k * t_val + eps)
            result += (A * j0(k * rho_val) + B * y0(k * rho_val)) * cos_term
        return result

    def x_1(self, rho_val: float, t_val: float) -> float:
        """Evaluate dx/drho at (rho, t)."""
        result = 0.0
        for k, A, B, eps in self.modes:
            cos_term = np.cos(k * t_val + eps)
            # d/drho[J_0(k*rho)] = -k * J_1(k*rho)
            # d/drho[Y_0(k*rho)] = -k * Y_1(k*rho)
            result += (-A * k * j1(k * rho_val) - B * k * y1(k * rho_val)) * cos_term
        return result

    def x_4(self, rho_val: float, t_val: float) -> float:
        """Evaluate dx/dt at (rho, t)."""
        result = 0.0
        for k, A, B, eps in self.modes:
            sin_term = -np.sin(k * t_val + eps)
            result += (A * j0(k * rho_val) + B * y0(k * rho_val)) * k * sin_term
        return result

    def verify_wave_equation(self, rho_val: float, t_val: float) -> float:
        """
        Verify that x satisfies the wave equation.

        Returns the residual: x₁₁ - x₄₄ + x₁/ρ (should be ~0).
        """
        eps = 1e-6

        # Numerical second derivatives
        x_11 = (self.x(rho_val + eps, t_val) - 2*self.x(rho_val, t_val) +
                self.x(rho_val - eps, t_val)) / eps**2
        x_44 = (self.x(rho_val, t_val + eps) - 2*self.x(rho_val, t_val) +
                self.x(rho_val, t_val - eps)) / eps**2
        x_1 = self.x_1(rho_val, t_val)

        return x_11 - x_44 + x_1 / rho_val


def create_standing_wave(k: float, A: float = 1.0) -> CylindricalWaveSolution:
    """
    Create a simple standing wave solution.

    x = A * J_0(k*rho) * cos(k*t)
    """
    return CylindricalWaveSolution(modes=[(k, A, 0.0, 0.0)])


def create_superposition(k_values: List[float],
                         A_values: List[float]) -> CylindricalWaveSolution:
    """
    Create a superposition of standing waves.

    x = sum_n A_n * J_0(k_n * rho) * cos(k_n * t)
    """
    modes = [(k, A, 0.0, 0.0) for k, A in zip(k_values, A_values)]
    return CylindricalWaveSolution(modes=modes)
