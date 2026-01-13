"""
Main Wave Conversion Class

Combines vacuum seed solutions with geodesic submanifolds to study
gravitational-electromagnetic wave mode conversion in cylindrical spacetimes.

Based on: Mishima & Tomizawa, Phys. Rev. D 110, 024038 (2024)

Usage:
    from em_cylindrical.wave_conversion import (
        WaveConversion, SolitonicSeed, ComplexLine, LagrangianPlane
    )

    # Create solution
    seed = SolitonicSeed.from_p_l(p=1.0, l=1.0)
    submanifold = LagrangianPlane()
    wc = WaveConversion(seed=seed, submanifold=submanifold)

    # Analyze conversion
    R_grav, R_em = wc.c_energy().occupancy_ratios(rho=2.0, t=1.0)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Callable
from .ernst import ErnstPotentials
from .seeds import VacuumSeed, LinearWaveSeed, SolitonicSeed
from .harmonic_map import GeodesicSubmanifold, ComplexLine, LagrangianPlane
from .c_energy import CEnergy, c_energy_from_ernst


class WaveConversion:
    """
    Main class for gravitational-electromagnetic wave mode conversion.

    Implements the harmonic mapping method from Mishima & Tomizawa (2024)
    for generating and analyzing Einstein-Maxwell solutions with
    cylindrical symmetry.

    The method works by:
    1. Starting with a vacuum seed solution (pure gravitational)
    2. Embedding into a geodesic submanifold of H²_C
    3. This "mixes" gravitational and electromagnetic components
    4. The result is an exact Einstein-Maxwell solution

    Key features:
    - Mode conversion between gravitational and electromagnetic waves
    - C-energy decomposition into 4 modes (+, ×, z, φ)
    - Conversion amplification ratios (0.4x to 2.4x for solitonic seeds)
    - Conversion to Misra-Radhakrishna variables for comparison
    """

    def __init__(self,
                 seed: VacuumSeed,
                 submanifold: GeodesicSubmanifold):
        """
        Initialize wave conversion solution.

        Args:
            seed: Vacuum seed solution (LinearWaveSeed or SolitonicSeed)
            submanifold: Geodesic submanifold (ComplexLine or LagrangianPlane)
        """
        self.seed = seed
        self.submanifold = submanifold

        # Cache for Ernst potentials
        self._ernst: Optional[ErnstPotentials] = None
        self._c_energy: Optional[CEnergy] = None

    def _get_xi_eta(self, rho: float, t: float) -> Tuple[complex, complex]:
        """Get (ξ, η) at a point using the harmonic map."""
        xi_v = self.seed.xi_v(rho, t)
        return self.submanifold.map_from_seed(xi_v)

    def xi(self, rho: float, t: float) -> complex:
        """Reduced gravitational potential ξ at (ρ, t)."""
        xi, _ = self._get_xi_eta(rho, t)
        return xi

    def eta(self, rho: float, t: float) -> complex:
        """Reduced electromagnetic potential η at (ρ, t)."""
        _, eta = self._get_xi_eta(rho, t)
        return eta

    def ernst_potentials(self, rho: float, t: float) -> Tuple[complex, complex]:
        """
        Get Ernst potentials (E, F) at a point.

        E = (1 + ξ) / (1 - ξ)  (gravitational)
        F = η / (1 - ξ)        (electromagnetic)

        Returns:
            (E, F) tuple of complex potentials
        """
        xi, eta = self._get_xi_eta(rho, t)

        if abs(1 - xi) < 1e-15:
            # Near singularity
            return (1e10 + 0j, 0j)

        E = (1 + xi) / (1 - xi)
        F = eta / (1 - xi)

        return (E, F)

    def psi(self, rho: float, t: float) -> float:
        """
        Metric function ψ at (ρ, t).

        From E = e^(2ψ) + |F|² - i*Φ, we have:
        e^(2ψ) = Re(E) - |F|²
        ψ = (1/2) * log(Re(E) - |F|²)
        """
        E, F = self.ernst_potentials(rho, t)

        exp_2psi = E.real - abs(F)**2
        if exp_2psi <= 0:
            # Degenerate case
            return 0.0

        return 0.5 * np.log(exp_2psi)

    def Phi(self, rho: float, t: float) -> float:
        """
        Twist potential Φ at (ρ, t).

        Φ = -Im(E)
        """
        E, _ = self.ernst_potentials(rho, t)
        return -E.imag

    def A_z(self, rho: float, t: float) -> float:
        """
        z-component of electromagnetic potential at (ρ, t).

        A_z = Re(F)
        """
        _, F = self.ernst_potentials(rho, t)
        return F.real

    def chi(self, rho: float, t: float) -> float:
        """
        Magnetic potential χ at (ρ, t).

        χ = Im(F)
        """
        _, F = self.ernst_potentials(rho, t)
        return F.imag

    def gamma(self, rho: float, t: float) -> float:
        """
        Metric function γ at (ρ, t).

        γ is related to C-energy by E = γ(ρ, t) - γ(0, t)

        For the harmonic map solutions, γ can be computed from:
        γ = ψ + (ρ/4) * [ψ_ρ² + ψ_t²] + ...

        This is an approximation; exact γ requires integration.
        """
        psi_val = self.psi(rho, t)
        eps = 1e-6

        # Derivatives of ψ
        psi_rho = (self.psi(rho + eps, t) - self.psi(rho - eps, t)) / (2 * eps)
        psi_t = (self.psi(rho, t + eps) - self.psi(rho, t - eps)) / (2 * eps)

        # Leading correction to γ from gravitational wave content
        gamma_approx = psi_val + (rho / 4) * (psi_rho**2 + psi_t**2)

        # Add electromagnetic corrections
        A_z_val = self.A_z(rho, t)
        chi_val = self.chi(rho, t)
        exp_2psi = np.exp(2 * psi_val)

        gamma_approx += 0.25 * (A_z_val**2 + chi_val**2) * exp_2psi

        return gamma_approx

    def metric_functions(self, rho: float, t: float) -> Dict[str, float]:
        """
        Get all metric and potential functions at a point.

        Returns dictionary with:
        - 'psi': Metric function ψ
        - 'gamma': Metric function γ
        - 'Phi': Twist potential Φ
        - 'A_z': EM z-potential
        - 'chi': Magnetic potential χ
        """
        return {
            'psi': self.psi(rho, t),
            'gamma': self.gamma(rho, t),
            'Phi': self.Phi(rho, t),
            'A_z': self.A_z(rho, t),
            'chi': self.chi(rho, t),
        }

    def electromagnetic_potential(self, rho: float, t: float) -> Tuple[float, float]:
        """
        Get electromagnetic potentials (A_z, χ) at a point.

        Returns:
            (A_z, chi) tuple
        """
        return (self.A_z(rho, t), self.chi(rho, t))

    def conversion_ratio(self) -> float:
        """
        Compute the electromagnetic conversion/amplification ratio.

        For the solitonic seed with Lagrangian plane, this gives the
        ratio of EM energy at future null infinity to past null infinity.

        Range: approximately 0.4 to 2.4 depending on parameters.

        Returns:
            EM amplification factor
        """
        if isinstance(self.seed, SolitonicSeed):
            return self.seed.compute_conversion_ratio()
        elif isinstance(self.submanifold, ComplexLine):
            # For complex line, the ratio is determined by theta
            return self.submanifold.electromagnetic_fraction()
        else:
            # Default: no conversion
            return 1.0

    def c_energy(self) -> CEnergy:
        """
        Get C-energy calculator for this solution.

        Returns:
            CEnergy instance configured for this solution
        """
        if self._c_energy is None:
            self._c_energy = CEnergy(
                psi_func=self.psi,
                gamma_func=self.gamma,
                A_z_func=self.A_z,
                chi_func=self.chi,
            )
        return self._c_energy

    def ernst(self) -> ErnstPotentials:
        """
        Get ErnstPotentials object for this solution.

        Returns:
            ErnstPotentials instance
        """
        if self._ernst is None:
            self._ernst = ErnstPotentials.from_reduced(
                xi_func=self.xi,
                eta_func=self.eta,
            )
        return self._ernst

    def to_misra_radhakrishna(self, rho: float, t: float) -> Dict[str, float]:
        """
        Convert to Misra-Radhakrishna variables for comparison.

        The Misra-Radhakrishna metric is:
            ds² = e^(λ-μ)(dt² - dρ²) - ρ²e^(-μ)dΦ² - e^μdz²

        Relation to Kompaneets-Jordan-Ehlers:
            e^μ = e^(2ψ)  →  μ = 2ψ
            e^(λ-μ) = e^(2(γ-ψ))  →  λ - μ = 2(γ - ψ)  →  λ = 2γ

        The electromagnetic potentials (φ, ψ_MR) in MR notation:
            ψ_MR = √(8π) * A_z  (z-component of 4-potential)
            φ_MR = √(8π) * χ    (azimuthal component)

        Returns dictionary with:
        - 'mu': Metric function μ
        - 'lambda': Metric function λ
        - 'phi': EM potential φ (azimuthal)
        - 'psi_MR': EM potential ψ (axial) - named to distinguish from metric ψ
        """
        psi_val = self.psi(rho, t)
        gamma_val = self.gamma(rho, t)
        A_z_val = self.A_z(rho, t)
        chi_val = self.chi(rho, t)

        # Coordinate conversion
        mu = 2 * psi_val
        lam = 2 * gamma_val

        # EM potential conversion
        sqrt_8pi = np.sqrt(8 * np.pi)
        psi_MR = sqrt_8pi * A_z_val
        phi_MR = sqrt_8pi * chi_val

        return {
            'mu': mu,
            'lambda': lam,
            'phi': phi_MR,
            'psi_MR': psi_MR,
        }

    def verify_constraint(self, rho: float, t: float,
                          tolerance: float = 1e-10) -> bool:
        """
        Verify |ξ|² + |η|² < 1 constraint.

        This constraint must be satisfied for valid solutions.
        """
        xi, eta = self._get_xi_eta(rho, t)
        return abs(xi)**2 + abs(eta)**2 < 1.0 + tolerance

    def constraint_value(self, rho: float, t: float) -> float:
        """
        Compute |ξ|² + |η|² at a point.

        Should be < 1 for valid solutions.
        """
        xi, eta = self._get_xi_eta(rho, t)
        return abs(xi)**2 + abs(eta)**2

    def verify_ernst_equations(self, rho: float, t: float,
                               eps: float = 1e-6) -> Dict[str, float]:
        """
        Check Ernst equation residuals at a point.

        Returns dictionary with residuals and constraint info.
        """
        return self.ernst().verify_ernst_equations(rho, t, eps)

    def __repr__(self) -> str:
        return f"WaveConversion(seed={self.seed}, submanifold={self.submanifold})"


def create_conversion_demo(case: str = 'solitonic_lagrangian'
                           ) -> WaveConversion:
    """
    Create a pre-configured WaveConversion for demonstration.

    Available cases:
    - 'solitonic_lagrangian': Solitonic seed with Lagrangian plane
                              (shows nontrivial conversion)
    - 'solitonic_complex': Solitonic seed with complex line
    - 'linear_complex': Linear wave seed with complex line
    - 'linear_lagrangian': Linear wave seed with Lagrangian plane

    Args:
        case: Name of the configuration

    Returns:
        WaveConversion instance
    """
    if case == 'solitonic_lagrangian':
        # Parameters satisfying q² - p² - l² = 1
        # p=1, l=1 → q = √3 ≈ 1.732
        seed = SolitonicSeed.from_p_l(p=1.0, l=1.0)
        submanifold = LagrangianPlane()

    elif case == 'solitonic_complex':
        seed = SolitonicSeed.from_p_l(p=1.0, l=1.0)
        submanifold = ComplexLine(theta=np.pi / 8)  # 22.5 degrees

    elif case == 'linear_complex':
        seed = LinearWaveSeed(amplitude=0.5, wavenumber=1.0)
        submanifold = ComplexLine(theta=np.pi / 6)  # 30 degrees

    elif case == 'linear_lagrangian':
        seed = LinearWaveSeed(amplitude=0.5, wavenumber=1.0)
        submanifold = LagrangianPlane()

    else:
        raise ValueError(f"Unknown case: {case}. "
                         f"Available: solitonic_lagrangian, solitonic_complex, "
                         f"linear_complex, linear_lagrangian")

    return WaveConversion(seed=seed, submanifold=submanifold)
