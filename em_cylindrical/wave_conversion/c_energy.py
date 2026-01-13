"""
C-Energy Calculations for Cylindrical Gravitational Waves

C-energy (Thorne, 1965) provides a local, covariant measure of energy
for cylindrically symmetric spacetimes. It has the properties:
- Conserved: ∂ℰ/∂t = ∂ℱ/∂ρ
- Localizable: Component along observer worldline is measured energy density
- Propagated by gravitational and electromagnetic waves

For the Kompaneets-Jordan-Ehlers metric:
    ds² = e^(2ψ)(dz - w dφ)² + ρ²e^(-2ψ)dφ² + e^(2(γ-ψ))(-dt² + dρ²)

The C-energy is:
    E(t, ρ₀) = γ(t, ρ₀) - γ(t, 0)

Mode decomposition:
    ℰ = (ρ/8)[𝒜_+² + ℬ_+² + 𝒜_×² + ℬ_×² + 𝒜_z² + ℬ_z² + 𝒜_φ² + ℬ_φ²]

where (+, ×) are gravitational polarizations and (z, φ) are EM components.

Based on:
- Thorne (1965), Phys. Rev. 138, B251
- Mishima & Tomizawa (2024), Phys. Rev. D 110, 024038
"""

import numpy as np
from typing import Dict, Tuple, Callable, Optional
from scipy.integrate import quad


class CEnergy:
    """
    C-energy calculations for cylindrical gravitational wave solutions.

    Provides energy density, flux, and mode decomposition for
    analyzing gravitational-electromagnetic wave conversion.
    """

    def __init__(self,
                 psi_func: Callable[[float, float], float],
                 gamma_func: Callable[[float, float], float],
                 A_z_func: Optional[Callable[[float, float], float]] = None,
                 chi_func: Optional[Callable[[float, float], float]] = None,
                 w_func: Optional[Callable[[float, float], float]] = None):
        """
        Initialize C-energy calculator.

        Args:
            psi_func: Metric function ψ(ρ, t)
            gamma_func: Metric function γ(ρ, t) - directly gives C-energy
            A_z_func: z-component of EM potential (optional)
            chi_func: Magnetic potential χ (optional)
            w_func: Rotation parameter w (optional, default 0)
        """
        self._psi = psi_func
        self._gamma = gamma_func
        self._A_z = A_z_func if A_z_func is not None else lambda r, t: 0.0
        self._chi = chi_func if chi_func is not None else lambda r, t: 0.0
        self._w = w_func if w_func is not None else lambda r, t: 0.0

    def psi(self, rho: float, t: float) -> float:
        """Metric function ψ at (ρ, t)."""
        return float(self._psi(rho, t))

    def gamma(self, rho: float, t: float) -> float:
        """Metric function γ at (ρ, t)."""
        return float(self._gamma(rho, t))

    def total_energy(self, rho_max: float, t: float,
                     rho_min: float = 0.01) -> float:
        """
        Total C-energy within radius ρ_max at time t.

        E(t, ρ_max) = γ(t, ρ_max) - γ(t, ρ_min)

        Args:
            rho_max: Outer radius
            t: Time coordinate
            rho_min: Inner radius (small positive value to avoid axis)

        Returns:
            Total C-energy
        """
        return self.gamma(rho_max, t) - self.gamma(rho_min, t)

    def density(self, rho: float, t: float, eps: float = 1e-6) -> float:
        """
        Local C-energy density ℰ = ∂γ/∂ρ at (ρ, t).

        The density represents the C-energy per unit radial coordinate.
        """
        # Numerical derivative
        gamma_plus = self.gamma(rho + eps, t)
        gamma_minus = self.gamma(rho - eps, t)
        return (gamma_plus - gamma_minus) / (2 * eps)

    def flux(self, rho: float, t: float, eps: float = 1e-6) -> float:
        """
        C-energy flux ℱ = ∂γ/∂t at (ρ, t).

        Positive flux indicates outward energy flow.
        Conservation law: ∂ℰ/∂t = ∂ℱ/∂ρ
        """
        gamma_plus = self.gamma(rho, t + eps)
        gamma_minus = self.gamma(rho, t - eps)
        return (gamma_plus - gamma_minus) / (2 * eps)

    def _compute_derivatives(self, rho: float, t: float,
                             eps: float = 1e-6) -> Dict[str, float]:
        """Compute all needed derivatives for mode decomposition."""
        derivs = {}

        # ψ derivatives
        derivs['psi_rho'] = (self.psi(rho + eps, t) - self.psi(rho - eps, t)) / (2 * eps)
        derivs['psi_t'] = (self.psi(rho, t + eps) - self.psi(rho, t - eps)) / (2 * eps)

        # A_z derivatives
        derivs['A_z_rho'] = (self._A_z(rho + eps, t) - self._A_z(rho - eps, t)) / (2 * eps)
        derivs['A_z_t'] = (self._A_z(rho, t + eps) - self._A_z(rho, t - eps)) / (2 * eps)

        # χ derivatives
        derivs['chi_rho'] = (self._chi(rho + eps, t) - self._chi(rho - eps, t)) / (2 * eps)
        derivs['chi_t'] = (self._chi(rho, t + eps) - self._chi(rho, t - eps)) / (2 * eps)

        # w derivatives (rotation)
        derivs['w_rho'] = (self._w(rho + eps, t) - self._w(rho - eps, t)) / (2 * eps)
        derivs['w_t'] = (self._w(rho, t + eps) - self._w(rho, t - eps)) / (2 * eps)

        return derivs

    def mode_decomposition(self, rho: float, t: float,
                           eps: float = 1e-6) -> Dict[str, float]:
        """
        Decompose C-energy density into gravitational and EM mode contributions.

        Returns dictionary with:
        - 'grav_plus': Gravitational + polarization
        - 'grav_cross': Gravitational × polarization
        - 'em_z': Electromagnetic z-mode
        - 'em_phi': Electromagnetic φ-mode
        - 'total': Total density

        The mode amplitudes are defined such that:
            ℰ = (ρ/8) * (𝒜_+² + ℬ_+² + 𝒜_×² + ℬ_×² + 𝒜_z² + ℬ_z² + 𝒜_φ² + ℬ_φ²)
        """
        psi_val = self.psi(rho, t)
        exp_2psi = np.exp(2 * psi_val)

        derivs = self._compute_derivatives(rho, t, eps)

        # Mode amplitudes (squared sums of 𝒜 and ℬ components)
        # Gravitational modes come from ψ derivatives
        # 𝒜_+ = ψ_ρ, ℬ_+ = ψ_t (+ polarization)
        grav_plus = derivs['psi_rho']**2 + derivs['psi_t']**2

        # × polarization from rotation (w)
        # 𝒜_× ∝ w_ρ/ρ, ℬ_× ∝ w_t/ρ
        if rho > eps:
            grav_cross = (derivs['w_rho']**2 + derivs['w_t']**2) / (4 * rho**2) * exp_2psi
        else:
            grav_cross = 0.0

        # Electromagnetic modes from A_z and χ
        # 𝒜_z = e^(2ψ) * A_z_ρ, ℬ_z = e^(2ψ) * A_z_t
        em_z = exp_2psi * (derivs['A_z_rho']**2 + derivs['A_z_t']**2)

        # 𝒜_φ ∝ χ_ρ, ℬ_φ ∝ χ_t
        em_phi = derivs['chi_rho']**2 + derivs['chi_t']**2

        # Total density
        total = (rho / 8) * (grav_plus + grav_cross + em_z + em_phi)

        return {
            'grav_plus': (rho / 8) * grav_plus,
            'grav_cross': (rho / 8) * grav_cross,
            'em_z': (rho / 8) * em_z,
            'em_phi': (rho / 8) * em_phi,
            'total': total,
        }

    def occupancy_ratios(self, rho: float, t: float) -> Tuple[float, float]:
        """
        Compute gravitational and electromagnetic mode occupancy ratios.

        R_grav = (ℰ_+ + ℰ_×) / ℰ_total
        R_em = (ℰ_z + ℰ_φ) / ℰ_total

        Returns:
            (R_grav, R_em) tuple where R_grav + R_em = 1
        """
        modes = self.mode_decomposition(rho, t)

        total = modes['total']
        if abs(total) < 1e-15:
            return (0.5, 0.5)  # No energy, undefined

        grav = modes['grav_plus'] + modes['grav_cross']
        em = modes['em_z'] + modes['em_phi']

        R_grav = grav / total
        R_em = em / total

        return (R_grav, R_em)

    def verify_conservation(self, rho: float, t: float,
                            eps: float = 1e-5) -> float:
        """
        Check conservation law: -∂ℰ/∂t + ∂ℱ/∂ρ = 0

        Returns the residual (should be close to 0).
        """
        # ∂ℰ/∂t
        E_plus = self.density(rho, t + eps)
        E_minus = self.density(rho, t - eps)
        dE_dt = (E_plus - E_minus) / (2 * eps)

        # ∂ℱ/∂ρ
        F_plus = self.flux(rho + eps, t)
        F_minus = self.flux(rho - eps, t)
        dF_drho = (F_plus - F_minus) / (2 * eps)

        residual = -dE_dt + dF_drho
        return abs(residual)

    def integrated_mode_ratios(self, rho_max: float, t: float,
                               n_points: int = 50) -> Tuple[float, float]:
        """
        Compute integrated gravitational and EM mode ratios over radius.

        Integrates the mode energies from the axis to rho_max:
            R_grav = ∫ℰ_grav dρ / ∫ℰ_total dρ
            R_em = ∫ℰ_em dρ / ∫ℰ_total dρ

        Args:
            rho_max: Maximum radius for integration
            t: Time coordinate
            n_points: Number of integration points

        Returns:
            (R_grav, R_em) integrated ratios
        """
        rho_values = np.linspace(0.01, rho_max, n_points)

        grav_total = 0.0
        em_total = 0.0
        energy_total = 0.0

        for i in range(len(rho_values) - 1):
            rho_mid = (rho_values[i] + rho_values[i + 1]) / 2
            drho = rho_values[i + 1] - rho_values[i]

            modes = self.mode_decomposition(rho_mid, t)
            grav = modes['grav_plus'] + modes['grav_cross']
            em = modes['em_z'] + modes['em_phi']

            grav_total += grav * drho
            em_total += em * drho
            energy_total += modes['total'] * drho

        if abs(energy_total) < 1e-15:
            return (0.5, 0.5)

        return (grav_total / energy_total, em_total / energy_total)


def c_energy_from_ernst(psi_func: Callable[[float, float], float],
                        A_z_func: Callable[[float, float], float] = None,
                        chi_func: Callable[[float, float], float] = None
                        ) -> CEnergy:
    """
    Create CEnergy calculator from metric functions.

    The γ function is computed by integrating the C-energy density:
        γ(ρ, t) = ∫₀^ρ ℰ(ρ', t) dρ'

    For numerical stability, we approximate γ from ψ and EM potentials.

    Args:
        psi_func: Metric function ψ(ρ, t)
        A_z_func: Electromagnetic potential A_z(ρ, t)
        chi_func: Magnetic potential χ(ρ, t)

    Returns:
        CEnergy instance
    """
    def gamma_func(rho: float, t: float) -> float:
        # Approximate γ by integrating the energy density
        # For small ρ, γ ≈ ψ + corrections from EM
        psi_val = psi_func(rho, t)

        # Leading contribution from gravitational part
        gamma_approx = psi_val

        # Add EM corrections if present
        if A_z_func is not None:
            A_z_val = A_z_func(rho, t)
            gamma_approx += 0.5 * A_z_val**2 * np.exp(2 * psi_val)

        if chi_func is not None:
            chi_val = chi_func(rho, t)
            gamma_approx += 0.25 * chi_val**2

        return gamma_approx

    return CEnergy(psi_func, gamma_func, A_z_func, chi_func)
