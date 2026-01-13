"""
Electromagnetic Field Tensor F_μν

The field tensor is defined by equation (5):
    F_μν = A_μ,ν - A_ν,μ

where A_α is the electromagnetic 4-potential.

For the cylindrically symmetric case with 4-potential components
A_2 = φ/√(8π) and A_3 = ψ/√(8π), the non-vanishing components are
given by equation (9):
    F₁₂ = -φ₁/√(8π)  (φ derivative w.r.t. ρ)
    F₁₃ = -ψ₁/√(8π)  (ψ derivative w.r.t. ρ)
    F₂₄ = φ₄/√(8π)   (φ derivative w.r.t. t)
    F₃₄ = ψ₄/√(8π)   (ψ derivative w.r.t. t)
"""

import sympy as sp
import numpy as np
from typing import Tuple, Callable

# Coordinate symbols
rho, t = sp.symbols('rho t', real=True, positive=True)


class ElectromagneticFieldTensor:
    """
    Electromagnetic field tensor F_μν for cylindrically symmetric spacetimes.

    Coordinates: (x¹, x², x³, x⁴) = (ρ, Φ, z, t)
    Indices: 0=ρ, 1=Φ, 2=z, 3=t
    """

    def __init__(self, phi_expr: sp.Expr, psi_expr: sp.Expr):
        """
        Initialize field tensor from 4-potential components.

        Args:
            phi_expr: SymPy expression for φ(ρ, t)
            psi_expr: SymPy expression for ψ(ρ, t)
        """
        self.phi_symbolic = phi_expr
        self.psi_symbolic = psi_expr

        self._build_tensor()
        self._create_numerical_functions()

    def _build_tensor(self):
        """Build the symbolic field tensor."""
        sqrt_8pi = sp.sqrt(8 * sp.pi)

        # Derivatives
        phi_1 = sp.diff(self.phi_symbolic, rho)  # ∂φ/∂ρ
        phi_4 = sp.diff(self.phi_symbolic, t)    # ∂φ/∂t
        psi_1 = sp.diff(self.psi_symbolic, rho)  # ∂ψ/∂ρ
        psi_4 = sp.diff(self.psi_symbolic, t)    # ∂ψ/∂t

        # Store derivatives symbolically
        self.phi_1_symbolic = phi_1
        self.phi_4_symbolic = phi_4
        self.psi_1_symbolic = psi_1
        self.psi_4_symbolic = psi_4

        # Build F_μν (covariant, antisymmetric)
        # Non-zero components from eq (9):
        # F₁₂ = -φ₁/√(8π)  -> F[0,1]
        # F₁₃ = -ψ₁/√(8π)  -> F[0,2]
        # F₂₄ = φ₄/√(8π)   -> F[1,3]
        # F₃₄ = ψ₄/√(8π)   -> F[2,3]

        self.F_symbolic = sp.Matrix([
            [0, -phi_1/sqrt_8pi, -psi_1/sqrt_8pi, 0],
            [phi_1/sqrt_8pi, 0, 0, phi_4/sqrt_8pi],
            [psi_1/sqrt_8pi, 0, 0, psi_4/sqrt_8pi],
            [0, -phi_4/sqrt_8pi, -psi_4/sqrt_8pi, 0]
        ])

    def _create_numerical_functions(self):
        """Create numerical evaluation functions."""
        self._F_funcs = {}
        for i in range(4):
            for j in range(4):
                expr = self.F_symbolic[i, j]
                if expr != 0:
                    self._F_funcs[(i, j)] = sp.lambdify((rho, t), expr, 'numpy')

    def at(self, rho_val: float, t_val: float) -> np.ndarray:
        """
        Evaluate the field tensor numerically at a point.

        Args:
            rho_val: Radial coordinate value
            t_val: Time coordinate value

        Returns:
            4x4 numpy array of F_μν components
        """
        F = np.zeros((4, 4))
        for (i, j), func in self._F_funcs.items():
            F[i, j] = func(rho_val, t_val)
        return F

    def invariants(self, rho_val: float, t_val: float,
                   g_inv: np.ndarray) -> Tuple[float, float]:
        """
        Compute the electromagnetic invariants.

        I₁ = F_μν F^μν (scalar invariant)
        I₂ = F_μν *F^μν (pseudoscalar invariant)

        Args:
            rho_val, t_val: Point coordinates
            g_inv: Inverse metric g^μν at the point

        Returns:
            Tuple (I₁, I₂)
        """
        F = self.at(rho_val, t_val)

        # Raise indices: F^μν = g^μα g^νβ F_αβ
        F_upper = g_inv @ F @ g_inv.T

        # I₁ = F_μν F^μν
        I1 = np.sum(F * F_upper)

        # I₂ requires dual tensor *F^μν
        # For simplicity, compute using Levi-Civita symbol
        sqrt_neg_g = np.sqrt(-np.linalg.det(np.linalg.inv(g_inv)))
        dual_F = np.zeros((4, 4))
        eps = np.zeros((4, 4, 4, 4))
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        eps[i, j, k, l] = levi_civita_4d(i, j, k, l)

        for i in range(4):
            for j in range(4):
                for k in range(4):
                    for l in range(4):
                        dual_F[i, j] += 0.5 * eps[i, j, k, l] * F_upper[k, l] / sqrt_neg_g

        I2 = np.sum(F * dual_F)

        return I1, I2


def levi_civita_4d(i: int, j: int, k: int, l: int) -> int:
    """Compute the 4D Levi-Civita symbol ε_ijkl."""
    indices = [i, j, k, l]
    if len(set(indices)) != 4:
        return 0

    # Count inversions
    inversions = 0
    for a in range(4):
        for b in range(a + 1, 4):
            if indices[a] > indices[b]:
                inversions += 1

    return 1 if inversions % 2 == 0 else -1


def compute_field_tensor_from_solution(solution) -> ElectromagneticFieldTensor:
    """
    Create a field tensor from any solution object.

    Args:
        solution: A CaseISolution, CaseIISolution, or CaseIIISolution object

    Returns:
        ElectromagneticFieldTensor instance
    """
    return ElectromagneticFieldTensor(
        solution.phi_symbolic,
        solution.psi_symbolic
    )
