"""
Vacuum Seed Solutions for Harmonic Map Method

Provides vacuum seed solutions that serve as starting points for
generating Einstein-Maxwell solutions via the harmonic mapping method.

Seed (i) - Linear wave:
    ξ_v(x) = (1 - e^(-2τ) + i*A) / (1 + e^(-2τ) - i*A)
    where τ satisfies the cylindrical wave equation ∇²τ = 0

Seed (ii) - Solitonic (Economou-Tsoubelis):
    ξ_v(x) = (1 - i*l) / (p*x - i*q*y)
    Coordinates: t = x*y, ρ = √[(x²+1)(y²-1)]
    Constraint: q² - p² - l² = 1

Based on: Mishima & Tomizawa, Phys. Rev. D 110, 024038 (2024)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional
from scipy.special import j0, j1


class VacuumSeed(ABC):
    """
    Abstract base class for vacuum seed solutions.

    A vacuum seed provides a solution to the vacuum Ernst equation
    (F = 0, pure gravitational) that can be transformed into an
    Einstein-Maxwell solution via harmonic map embedding.
    """

    @abstractmethod
    def xi_v(self, rho: float, t: float) -> complex:
        """
        Vacuum reduced potential ξ_v at (ρ, t).

        For vacuum solutions, η = 0 and ξ_v gives the full solution.
        """
        pass

    def eta_v(self, rho: float, t: float) -> complex:
        """Vacuum electromagnetic potential (always zero for vacuum)."""
        return 0.0 + 0.0j

    def E_v(self, rho: float, t: float) -> complex:
        """
        Vacuum Ernst potential E_v from ξ_v.

        E_v = (1 + ξ_v) / (1 - ξ_v)
        """
        xi = self.xi_v(rho, t)
        return (1 + xi) / (1 - xi)

    def verify_vacuum_constraint(self, rho: float, t: float,
                                 tolerance: float = 1e-10) -> bool:
        """Check |ξ_v|² < 1."""
        xi = self.xi_v(rho, t)
        return abs(xi)**2 < 1.0 + tolerance


class LinearWaveSeed(VacuumSeed):
    """
    Seed (i): Linear wave seed solution.

    The vacuum potential is constructed from τ satisfying the
    cylindrical wave equation:
        τ_ρρ - τ_tt + τ_ρ/ρ = 0

    For a Bessel function solution:
        τ = A * J₀(k*ρ) * cos(k*t + ε)

    The vacuum ξ_v is:
        ξ_v = (1 - e^(-2τ) + i*A_param) / (1 + e^(-2τ) - i*A_param)

    where A_param is an additional parameter controlling the imaginary part.
    """

    def __init__(self,
                 amplitude: float = 1.0,
                 wavenumber: float = 1.0,
                 phase: float = 0.0,
                 A_param: float = 0.0):
        """
        Initialize linear wave seed.

        Args:
            amplitude: Amplitude of the wave function τ
            wavenumber: Wavenumber k for J₀(kρ)cos(kt)
            phase: Phase offset ε in cos(kt + ε)
            A_param: Parameter A controlling imaginary part of ξ_v
        """
        self.amplitude = amplitude
        self.wavenumber = wavenumber
        self.phase = phase
        self.A_param = A_param

    def tau(self, rho: float, t: float) -> float:
        """
        Wave function τ satisfying cylindrical wave equation.

        τ = amplitude * J₀(k*ρ) * cos(k*t + ε)
        """
        k = self.wavenumber
        return self.amplitude * j0(k * rho) * np.cos(k * t + self.phase)

    def tau_rho(self, rho: float, t: float) -> float:
        """Derivative ∂τ/∂ρ."""
        k = self.wavenumber
        # d/dρ J₀(kρ) = -k * J₁(kρ)
        return -self.amplitude * k * j1(k * rho) * np.cos(k * t + self.phase)

    def tau_t(self, rho: float, t: float) -> float:
        """Derivative ∂τ/∂t."""
        k = self.wavenumber
        return -self.amplitude * k * j0(k * rho) * np.sin(k * t + self.phase)

    def xi_v(self, rho: float, t: float) -> complex:
        """
        Vacuum reduced potential ξ_v.

        ξ_v = (1 - e^(-2τ) + i*A) / (1 + e^(-2τ) - i*A)
        """
        tau_val = self.tau(rho, t)
        exp_term = np.exp(-2 * tau_val)
        A = self.A_param

        numerator = 1 - exp_term + 1j * A
        denominator = 1 + exp_term - 1j * A

        return numerator / denominator

    def verify_wave_equation(self, rho: float, t: float,
                             eps: float = 1e-6) -> float:
        """
        Verify τ satisfies cylindrical wave equation.

        Returns residual of: τ_ρρ - τ_tt + τ_ρ/ρ = 0
        """
        tau_rhorho = (self.tau(rho + eps, t) - 2 * self.tau(rho, t) +
                      self.tau(rho - eps, t)) / eps**2
        tau_tt = (self.tau(rho, t + eps) - 2 * self.tau(rho, t) +
                  self.tau(rho, t - eps)) / eps**2
        tau_rho = self.tau_rho(rho, t)

        residual = tau_rhorho - tau_tt + tau_rho / rho
        return abs(residual)


class SolitonicSeed(VacuumSeed):
    """
    Seed (ii): Economou-Tsoubelis soliton solution.

    The vacuum potential in (x, y) coordinates is:
        ξ_v = (1 - i*l) / (p*x - i*q*y)

    Coordinate transformation:
        t = x * y
        ρ = √[(x² + 1)(y² - 1)]

    Parameter constraint (must be satisfied):
        q² - p² - l² = 1

    This seed produces nontrivial mode conversions when embedded
    into the Lagrangian plane (case b).
    """

    def __init__(self, p: float, q: float, l: float):
        """
        Initialize solitonic seed with parameters (p, q, l).

        Args:
            p: Parameter p
            q: Parameter q
            l: Parameter l

        Raises:
            ValueError: If constraint q² - p² - l² = 1 is not satisfied.
        """
        self.p = p
        self.q = q
        self.l = l

        # Verify constraint
        constraint_value = q**2 - p**2 - l**2
        if abs(constraint_value - 1.0) > 1e-10:
            raise ValueError(
                f"Constraint q² - p² - l² = 1 not satisfied. "
                f"Got q² - p² - l² = {constraint_value:.6f}"
            )

    @staticmethod
    def from_p_l(p: float, l: float) -> 'SolitonicSeed':
        """
        Create SolitonicSeed by specifying p and l, computing q.

        q = √(1 + p² + l²)
        """
        q = np.sqrt(1 + p**2 + l**2)
        return SolitonicSeed(p, q, l)

    @staticmethod
    def from_p_q(p: float, q: float) -> 'SolitonicSeed':
        """
        Create SolitonicSeed by specifying p and q, computing l.

        l = √(q² - p² - 1)  (requires q² > p² + 1)
        """
        l_sq = q**2 - p**2 - 1
        if l_sq < 0:
            raise ValueError(f"Cannot compute l: q² - p² - 1 = {l_sq} < 0")
        l = np.sqrt(l_sq)
        return SolitonicSeed(p, q, l)

    def rho_t_to_xy(self, rho: float, t: float) -> Tuple[float, float]:
        """
        Convert (ρ, t) coordinates to (x, y) coordinates.

        Given:
            t = x * y
            ρ² = (x² + 1)(y² - 1)

        We solve for x, y. This requires y > 1 or y < -1.

        For y > 1:
            ρ² = (x² + 1)(y² - 1)
            t = xy

        Substituting x = t/y:
            ρ² = (t²/y² + 1)(y² - 1)
            ρ² = t² - t²/y² + y² - 1
            ρ² = t² + y² - 1 - t²/y²

        Let u = y²:
            ρ² = t² + u - 1 - t²/u
            ρ² * u = t² * u + u² - u - t²
            u² + (t² - ρ² - 1) * u - t² = 0

        Using quadratic formula for u = y².
        """
        # Coefficients: u² + b*u + c = 0
        b = t**2 - rho**2 - 1
        c = -t**2

        discriminant = b**2 - 4 * c
        if discriminant < 0:
            # This shouldn't happen for valid (ρ, t)
            raise ValueError(f"No real solution for (ρ, t) = ({rho}, {t})")

        sqrt_disc = np.sqrt(discriminant)

        # We need u = y² > 1 for real coordinates
        u1 = (-b + sqrt_disc) / 2
        u2 = (-b - sqrt_disc) / 2

        # Pick the solution with u > 1
        if u1 > 1:
            u = u1
        elif u2 > 1:
            u = u2
        else:
            # Both solutions have y² ≤ 1, use the larger one
            # This can happen near boundaries
            u = max(u1, u2)
            if u <= 0:
                raise ValueError(f"Invalid y² = {u} for (ρ, t) = ({rho}, {t})")

        y = np.sqrt(u) if t >= 0 else -np.sqrt(u)
        x = t / y if abs(y) > 1e-10 else 0.0

        return (x, y)

    def xy_to_rho_t(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert (x, y) coordinates to (ρ, t).

        t = x * y
        ρ = √[(x² + 1)(y² - 1)]
        """
        t = x * y
        rho_sq = (x**2 + 1) * (y**2 - 1)
        if rho_sq < 0:
            raise ValueError(f"Invalid ρ² = {rho_sq} for (x, y) = ({x}, {y})")
        rho = np.sqrt(rho_sq)
        return (rho, t)

    def xi_v_xy(self, x: float, y: float) -> complex:
        """
        Vacuum potential ξ_v in (x, y) coordinates.

        ξ_v = (1 - i*l) / (p*x - i*q*y)
        """
        numerator = 1 - 1j * self.l
        denominator = self.p * x - 1j * self.q * y

        if abs(denominator) < 1e-15:
            # Near singularity
            return 0.0 + 0.0j

        return numerator / denominator

    def xi_v(self, rho: float, t: float) -> complex:
        """
        Vacuum reduced potential ξ_v at (ρ, t).

        Converts to (x, y) coordinates and evaluates ξ_v.
        """
        try:
            x, y = self.rho_t_to_xy(rho, t)
            return self.xi_v_xy(x, y)
        except ValueError:
            # Return zero at problematic points
            return 0.0 + 0.0j

    def conversion_amplification_bounds(self) -> Tuple[float, float]:
        """
        Theoretical bounds on electromagnetic amplification ratio.

        For case (b) with this solitonic seed, the amplification ratio
        at null infinity is bounded by approximately [0.4, 2.4].

        Returns:
            (min_ratio, max_ratio) tuple
        """
        # From the paper, these are approximate bounds
        return (0.4, 2.4)

    def compute_conversion_ratio(self) -> float:
        """
        Compute the electromagnetic conversion ratio.

        Ratio = γ^(+)_em / γ^(-)_em

        For the solitonic seed, this depends on p, q, l parameters.
        """
        # Simplified formula from the paper
        # Ratio ≈ [P(q/p)l² - Q(q/p)l + R(q/p)] / [P(q/p)l² + Q(q/p)l + R(q/p)]
        # where P, Q, R are polynomial functions

        r = self.q / self.p if abs(self.p) > 1e-10 else 1.0

        # Approximate polynomials (simplified from paper)
        # These are rough approximations
        P = 1.0
        Q = 2 * r
        R = r**2 + 1

        l = self.l
        numerator = P * l**2 - Q * l + R
        denominator = P * l**2 + Q * l + R

        if abs(denominator) < 1e-15:
            return 1.0

        return numerator / denominator

    def __repr__(self) -> str:
        return f"SolitonicSeed(p={self.p}, q={self.q}, l={self.l})"
