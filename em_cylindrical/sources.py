"""
Non-Vacuum Solutions with Current Sources

Extends the framework to handle solutions with J^μ ≠ 0.

Maxwell's equations with sources:
    ∇_μ F^μν = 4π J^ν

Given a current profile J^μ(ρ, t), this module:
1. Verifies consistency with Maxwell equations
2. Provides predefined current profiles (uniform, surface, etc.)
3. Allows specification of custom current distributions
"""

import sympy as sp
import numpy as np
from typing import Callable, Dict, Tuple, Optional, Union
from abc import ABC, abstractmethod

# Symbols for symbolic computation
rho_sym, t_sym = sp.symbols('rho t', real=True, positive=True)


class CurrentProfile(ABC):
    """Abstract base class for current density profiles."""

    @abstractmethod
    def J_rho(self, rho_val: float, t_val: float) -> float:
        """Radial current density component."""
        pass

    @abstractmethod
    def J_Phi(self, rho_val: float, t_val: float) -> float:
        """Azimuthal current density component."""
        pass

    @abstractmethod
    def J_z(self, rho_val: float, t_val: float) -> float:
        """Axial current density component."""
        pass

    @abstractmethod
    def J_t(self, rho_val: float, t_val: float) -> float:
        """Charge density (timelike component)."""
        pass

    def J_4vector(self, rho_val: float, t_val: float) -> np.ndarray:
        """Return full 4-current vector."""
        return np.array([
            self.J_rho(rho_val, t_val),
            self.J_Phi(rho_val, t_val),
            self.J_z(rho_val, t_val),
            self.J_t(rho_val, t_val)
        ])


class UniformAxialCurrent(CurrentProfile):
    """
    Uniform axial current distribution inside a cylinder.

    J^z = I_0 / (π R²)  for ρ < R
    J^z = 0             for ρ ≥ R

    This represents a wire or conductor with uniform current density.
    """

    def __init__(self, I_0: float, R: float):
        """
        Initialize uniform axial current.

        Args:
            I_0: Total current (in geometric units)
            R: Radius of current-carrying region
        """
        self.I_0 = I_0
        self.R = R
        self.J_z_uniform = I_0 / (np.pi * R**2)

    def J_rho(self, rho_val: float, t_val: float) -> float:
        return 0.0

    def J_Phi(self, rho_val: float, t_val: float) -> float:
        return 0.0

    def J_z(self, rho_val: float, t_val: float) -> float:
        if rho_val < self.R:
            return self.J_z_uniform
        return 0.0

    def J_t(self, rho_val: float, t_val: float) -> float:
        return 0.0  # No charge density for steady current


class SurfaceCurrent(CurrentProfile):
    """
    Surface current on a cylindrical shell at ρ = R.

    Approximated as a thin shell of width δ:
    J^z = K_z / δ  for |ρ - R| < δ/2
    J^z = 0        otherwise

    In the limit δ → 0, this approaches K_z δ(ρ - R).
    """

    def __init__(self, K_z: float, R: float, delta: float = 0.01):
        """
        Initialize surface current.

        Args:
            K_z: Surface current density (current per unit length)
            R: Radius of cylindrical surface
            delta: Thickness of approximation shell
        """
        self.K_z = K_z
        self.R = R
        self.delta = delta

    def J_rho(self, rho_val: float, t_val: float) -> float:
        return 0.0

    def J_Phi(self, rho_val: float, t_val: float) -> float:
        return 0.0

    def J_z(self, rho_val: float, t_val: float) -> float:
        if abs(rho_val - self.R) < self.delta / 2:
            return self.K_z / self.delta
        return 0.0

    def J_t(self, rho_val: float, t_val: float) -> float:
        return 0.0


class CustomCurrent(CurrentProfile):
    """
    Custom current profile specified by functions.
    """

    def __init__(self,
                 J_rho_func: Callable[[float, float], float] = None,
                 J_Phi_func: Callable[[float, float], float] = None,
                 J_z_func: Callable[[float, float], float] = None,
                 J_t_func: Callable[[float, float], float] = None):
        """
        Initialize with custom functions.

        Args:
            J_rho_func: Function (ρ, t) → J^ρ
            J_Phi_func: Function (ρ, t) → J^Φ
            J_z_func: Function (ρ, t) → J^z
            J_t_func: Function (ρ, t) → J^t (charge density)
        """
        self._J_rho = J_rho_func or (lambda r, t: 0.0)
        self._J_Phi = J_Phi_func or (lambda r, t: 0.0)
        self._J_z = J_z_func or (lambda r, t: 0.0)
        self._J_t = J_t_func or (lambda r, t: 0.0)

    def J_rho(self, rho_val: float, t_val: float) -> float:
        return self._J_rho(rho_val, t_val)

    def J_Phi(self, rho_val: float, t_val: float) -> float:
        return self._J_Phi(rho_val, t_val)

    def J_z(self, rho_val: float, t_val: float) -> float:
        return self._J_z(rho_val, t_val)

    def J_t(self, rho_val: float, t_val: float) -> float:
        return self._J_t(rho_val, t_val)


class SourcedSolution:
    """
    Container for electromagnetic solutions with current sources.

    Given:
    - A current profile J^μ(ρ, t)
    - Metric functions λ(ρ, t) and μ(ρ, t)
    - Potential functions φ(ρ, t) and ψ(ρ, t)

    Verifies Maxwell's equations:
        ∇_μ F^μν = 4π J^ν
    """

    def __init__(self, current_profile: CurrentProfile,
                 phi_func: Callable[[float, float], float] = None,
                 psi_func: Callable[[float, float], float] = None,
                 mu_func: Callable[[float, float], float] = None,
                 lambda_func: Callable[[float, float], float] = None):
        """
        Initialize sourced solution.

        Args:
            current_profile: CurrentProfile object specifying J^μ
            phi_func: Function (ρ, t) → φ (electromagnetic potential)
            psi_func: Function (ρ, t) → ψ (electromagnetic potential)
            mu_func: Function (ρ, t) → μ (metric function)
            lambda_func: Function (ρ, t) → λ (metric function)
        """
        self.current = current_profile

        # Default to flat spacetime if not specified
        self._phi = phi_func or (lambda r, t: 0.0)
        self._psi = psi_func or (lambda r, t: 0.0)
        self._mu = mu_func or (lambda r, t: 0.0)
        self._lambda = lambda_func or (lambda r, t: 0.0)

    def phi(self, rho_val: float, t_val: float) -> float:
        return self._phi(rho_val, t_val)

    def psi(self, rho_val: float, t_val: float) -> float:
        return self._psi(rho_val, t_val)

    def mu(self, rho_val: float, t_val: float) -> float:
        return self._mu(rho_val, t_val)

    def lambda_(self, rho_val: float, t_val: float) -> float:
        return self._lambda(rho_val, t_val)

    def field_tensor_at(self, rho_val: float, t_val: float) -> np.ndarray:
        """Compute F_μν from the potentials."""
        eps = 1e-6

        phi_1 = (self.phi(rho_val + eps, t_val) - self.phi(rho_val - eps, t_val)) / (2*eps)
        phi_4 = (self.phi(rho_val, t_val + eps) - self.phi(rho_val, t_val - eps)) / (2*eps)
        psi_1 = (self.psi(rho_val + eps, t_val) - self.psi(rho_val - eps, t_val)) / (2*eps)
        psi_4 = (self.psi(rho_val, t_val + eps) - self.psi(rho_val, t_val - eps)) / (2*eps)

        sqrt_8pi = np.sqrt(8 * np.pi)

        F = np.zeros((4, 4))
        F[0, 1] = -phi_1 / sqrt_8pi
        F[1, 0] = -F[0, 1]
        F[0, 2] = -psi_1 / sqrt_8pi
        F[2, 0] = -F[0, 2]
        F[1, 3] = phi_4 / sqrt_8pi
        F[3, 1] = -F[1, 3]
        F[2, 3] = psi_4 / sqrt_8pi
        F[3, 2] = -F[2, 3]

        return F

    def verify_maxwell_with_source(self, rho_val: float, t_val: float) -> Dict[str, float]:
        """
        Verify ∇_μ F^μν = 4π J^ν at a point.

        Returns residuals for each component.
        """
        from .current import CurrentDensity

        # Use CurrentDensity to compute divergence
        # We need to make this object look like a solution
        class TempSolution:
            def __init__(self, sourced):
                self.sourced = sourced

            def phi(self, r, t):
                return self.sourced.phi(r, t)

            def psi(self, r, t):
                return self.sourced.psi(r, t)

            def mu(self, r, t):
                return self.sourced.mu(r, t)

            def lambda_(self, r, t):
                return self.sourced.lambda_(r, t)

            def field_tensor_at(self, r, t):
                return self.sourced.field_tensor_at(r, t)

        temp = TempSolution(self)
        current_calc = CurrentDensity(temp)

        # Compute ∇_μ F^μν / 4π (what the current should be)
        J_computed = current_calc.current_4vector_at(rho_val, t_val)

        # Get specified current
        J_specified = self.current.J_4vector(rho_val, t_val)

        return {
            'J_rho_residual': J_computed[0] - J_specified[0],
            'J_Phi_residual': J_computed[1] - J_specified[1],
            'J_z_residual': J_computed[2] - J_specified[2],
            'J_t_residual': J_computed[3] - J_specified[3],
            'J_computed': J_computed,
            'J_specified': J_specified,
        }


# Convenience functions for common configurations

def uniform_axial_current(I_0: float, R: float = 1.0) -> SourcedSolution:
    """
    Create a sourced solution with uniform axial current.

    This is a simplified model - the potentials would need to be
    solved from Maxwell's equations for full consistency.

    Args:
        I_0: Total axial current
        R: Radius of current-carrying region

    Returns:
        SourcedSolution with uniform J^z inside ρ < R
    """
    current = UniformAxialCurrent(I_0, R)

    # For a uniform current in flat spacetime (μ = λ = 0):
    # The magnetic field is B_Φ = (2I_0/c)(ρ/R²) inside
    # This corresponds to ψ ∝ ρ² inside the cylinder

    def psi_func(rho, t):
        if rho < R:
            # ψ for uniform current (simplified, flat spacetime)
            return I_0 * rho**2 / (2 * R**2)
        else:
            # Outside: ψ = I_0 * (1/2 + log(ρ/R)) approximately
            return I_0 * (0.5 + np.log(rho / R))

    return SourcedSolution(
        current_profile=current,
        psi_func=psi_func,
    )


def surface_current_cylinder(K_z: float, R: float = 1.0,
                              delta: float = 0.01) -> SourcedSolution:
    """
    Create a sourced solution with surface current on a cylinder.

    Args:
        K_z: Surface current density (current per unit length)
        R: Cylinder radius
        delta: Shell thickness for approximation

    Returns:
        SourcedSolution with K_z at ρ = R
    """
    current = SurfaceCurrent(K_z, R, delta)

    # For a surface current at ρ = R in flat spacetime:
    # Inside: B = 0 (no enclosed current)
    # Outside: B_Φ = 2K_z/ρ

    def psi_func(rho, t):
        if rho < R - delta/2:
            return 0.0  # No field inside
        elif rho > R + delta/2:
            return 2 * K_z * np.log(rho / R)  # Outside the surface
        else:
            # Interpolate in the shell
            return K_z * (rho - R + delta/2) / delta * np.log(rho / R)

    return SourcedSolution(
        current_profile=current,
        psi_func=psi_func,
    )
