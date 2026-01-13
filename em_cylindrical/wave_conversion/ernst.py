"""
Ernst Potentials for Cylindrical Einstein-Maxwell Systems

The Ernst formulation uses complex potentials to encode gravitational
and electromagnetic degrees of freedom:

    E = e^(2ψ) + |F|² - i*Φ    (gravitational Ernst potential)
    F = A_z + i*χ               (electromagnetic Ernst potential)

Reduced variables (unit disc coordinates):
    ξ = (E - 1) / (E + 1)
    η = 2*F / (E + 1)

Constraint: |ξ|² + |η|² < 1

Ernst equations:
    (Re[E] - |F|²) ∇²E = (∇E - 2*F̄*∇F) · ∇E
    (Re[E] - |F|²) ∇²F = (∇E - 2*F̄*∇F) · ∇F

Based on: Ernst (1974), Mishima & Tomizawa (2024)
"""

import numpy as np
from typing import Callable, Tuple, Dict, Optional


class ErnstPotentials:
    """
    Complex Ernst potentials E and F for cylindrical Einstein-Maxwell systems.

    The Ernst formulation provides an elegant way to encode both gravitational
    and electromagnetic fields using complex potentials. This is particularly
    useful for generating new solutions via transformation methods.
    """

    def __init__(self,
                 psi_func: Callable[[float, float], float],
                 Phi_func: Optional[Callable[[float, float], float]] = None,
                 A_z_func: Optional[Callable[[float, float], float]] = None,
                 chi_func: Optional[Callable[[float, float], float]] = None):
        """
        Initialize Ernst potentials from metric and EM potential functions.

        Args:
            psi_func: Metric function ψ(ρ, t) - determines g_zz = e^(2ψ)
            Phi_func: Twist potential Φ(ρ, t) - imaginary part of E
            A_z_func: z-component of 4-potential A_z(ρ, t)
            chi_func: Magnetic potential χ(ρ, t) - imaginary part of F

        If Phi_func, A_z_func, or chi_func are None, they default to zero.
        """
        self._psi = psi_func
        self._Phi = Phi_func if Phi_func is not None else lambda r, t: 0.0
        self._A_z = A_z_func if A_z_func is not None else lambda r, t: 0.0
        self._chi = chi_func if chi_func is not None else lambda r, t: 0.0

    def psi(self, rho: float, t: float) -> float:
        """Metric function ψ at (ρ, t)."""
        return float(self._psi(rho, t))

    def Phi(self, rho: float, t: float) -> float:
        """Twist potential Φ at (ρ, t)."""
        return float(self._Phi(rho, t))

    def A_z(self, rho: float, t: float) -> float:
        """z-component of electromagnetic potential at (ρ, t)."""
        return float(self._A_z(rho, t))

    def chi(self, rho: float, t: float) -> float:
        """Magnetic potential χ at (ρ, t)."""
        return float(self._chi(rho, t))

    def E(self, rho: float, t: float) -> complex:
        """
        Gravitational Ernst potential E at (ρ, t).

        E = e^(2ψ) + |F|² - i*Φ
        """
        psi_val = self.psi(rho, t)
        Phi_val = self.Phi(rho, t)
        F_val = self.F(rho, t)

        return np.exp(2 * psi_val) + abs(F_val)**2 - 1j * Phi_val

    def F(self, rho: float, t: float) -> complex:
        """
        Electromagnetic Ernst potential F at (ρ, t).

        F = A_z + i*χ
        """
        return self._A_z(rho, t) + 1j * self._chi(rho, t)

    def xi(self, rho: float, t: float) -> complex:
        """
        Reduced gravitational potential ξ at (ρ, t).

        ξ = (E - 1) / (E + 1)

        Maps E to the unit disc (Poincaré model).
        """
        E_val = self.E(rho, t)
        return (E_val - 1) / (E_val + 1)

    def eta(self, rho: float, t: float) -> complex:
        """
        Reduced electromagnetic potential η at (ρ, t).

        η = 2*F / (E + 1)
        """
        E_val = self.E(rho, t)
        F_val = self.F(rho, t)
        return 2 * F_val / (E_val + 1)

    def constraint_value(self, rho: float, t: float) -> float:
        """
        Check the constraint |ξ|² + |η|² < 1.

        Returns the value |ξ|² + |η|². Should be < 1 for valid solutions.
        """
        xi_val = self.xi(rho, t)
        eta_val = self.eta(rho, t)
        return abs(xi_val)**2 + abs(eta_val)**2

    def verify_constraint(self, rho: float, t: float, tolerance: float = 1e-10) -> bool:
        """
        Verify that the constraint |ξ|² + |η|² < 1 is satisfied.

        Args:
            rho, t: Point coordinates
            tolerance: How much above 1 is allowed (for numerical errors)

        Returns:
            True if constraint is satisfied (within tolerance)
        """
        return self.constraint_value(rho, t) < 1.0 + tolerance

    def verify_ernst_equations(self, rho: float, t: float,
                               eps: float = 1e-6) -> Dict[str, float]:
        """
        Check how well the Ernst equations are satisfied at a point.

        Ernst equations in reduced form:
            (|ξ|² + |η|² - 1) ∇²ξ = 2*(ξ̄*∇ξ + η̄*∇η) · ∇ξ
            (|ξ|² + |η|² - 1) ∇²η = 2*(ξ̄*∇ξ + η̄*∇η) · ∇η

        Returns dictionary of residuals (should be close to 0).
        """
        # Get xi and eta at point and nearby
        xi_0 = self.xi(rho, t)
        eta_0 = self.eta(rho, t)

        # First derivatives (central differences)
        xi_rho = (self.xi(rho + eps, t) - self.xi(rho - eps, t)) / (2 * eps)
        xi_t = (self.xi(rho, t + eps) - self.xi(rho, t - eps)) / (2 * eps)
        eta_rho = (self.eta(rho + eps, t) - self.eta(rho - eps, t)) / (2 * eps)
        eta_t = (self.eta(rho, t + eps) - self.eta(rho, t - eps)) / (2 * eps)

        # Second derivatives
        xi_rhorho = (self.xi(rho + eps, t) - 2 * xi_0 + self.xi(rho - eps, t)) / eps**2
        xi_tt = (self.xi(rho, t + eps) - 2 * xi_0 + self.xi(rho, t - eps)) / eps**2
        eta_rhorho = (self.eta(rho + eps, t) - 2 * eta_0 + self.eta(rho - eps, t)) / eps**2
        eta_tt = (self.eta(rho, t + eps) - 2 * eta_0 + self.eta(rho, t - eps)) / eps**2

        # Cylindrical Laplacian: ∇² = ∂²/∂ρ² + (1/ρ)∂/∂ρ - ∂²/∂t²
        laplacian_xi = xi_rhorho + xi_rho / rho - xi_tt
        laplacian_eta = eta_rhorho + eta_rho / rho - eta_tt

        # Constraint factor
        constraint = abs(xi_0)**2 + abs(eta_0)**2 - 1

        # LHS of Ernst equations
        lhs_xi = constraint * laplacian_xi
        lhs_eta = constraint * laplacian_eta

        # Gradient dot products for RHS
        # (ξ̄*∇ξ + η̄*∇η) in cylindrical: need (ρ, t) components
        # Using flat metric signature (-,+) for (t, ρ)
        grad_factor_rho = np.conj(xi_0) * xi_rho + np.conj(eta_0) * eta_rho
        grad_factor_t = np.conj(xi_0) * xi_t + np.conj(eta_0) * eta_t

        # ∇ξ · (factor) with metric: -dt² + dρ²
        # So dot product: -factor_t * xi_t + factor_rho * xi_rho
        rhs_xi = 2 * (-grad_factor_t * xi_t + grad_factor_rho * xi_rho)
        rhs_eta = 2 * (-grad_factor_t * eta_t + grad_factor_rho * eta_rho)

        # Residuals
        residual_xi = abs(lhs_xi - rhs_xi)
        residual_eta = abs(lhs_eta - rhs_eta)

        return {
            'ernst_xi_residual': residual_xi,
            'ernst_eta_residual': residual_eta,
            'constraint_value': abs(xi_0)**2 + abs(eta_0)**2,
            'constraint_satisfied': self.verify_constraint(rho, t),
        }

    @classmethod
    def from_reduced(cls, xi_func: Callable[[float, float], complex],
                     eta_func: Callable[[float, float], complex]) -> 'ErnstPotentials':
        """
        Create ErnstPotentials from reduced potentials ξ and η.

        This inverts the transformations:
            E = (1 + ξ) / (1 - ξ)
            F = η * (E + 1) / 2 = η / (1 - ξ)

        Then extracts ψ, Φ, A_z, χ from E and F.
        """
        def psi_func(rho, t):
            xi = xi_func(rho, t)
            eta = eta_func(rho, t)
            E = (1 + xi) / (1 - xi)
            F = eta / (1 - xi)
            # E = e^(2ψ) + |F|² - i*Φ
            # Re(E) = e^(2ψ) + |F|²
            re_E = E.real
            F_mag_sq = abs(F)**2
            exp_2psi = re_E - F_mag_sq
            if exp_2psi <= 0:
                return 0.0  # Degenerate case
            return 0.5 * np.log(exp_2psi)

        def Phi_func(rho, t):
            xi = xi_func(rho, t)
            E = (1 + xi) / (1 - xi)
            # Im(E) = -Φ
            return -E.imag

        def A_z_func(rho, t):
            xi = xi_func(rho, t)
            eta = eta_func(rho, t)
            F = eta / (1 - xi)
            return F.real

        def chi_func(rho, t):
            xi = xi_func(rho, t)
            eta = eta_func(rho, t)
            F = eta / (1 - xi)
            return F.imag

        return cls(psi_func, Phi_func, A_z_func, chi_func)


def ernst_from_misra_radhakrishna(mu_func: Callable[[float, float], float],
                                  psi_MR_func: Callable[[float, float], float],
                                  phi_func: Callable[[float, float], float] = None
                                  ) -> ErnstPotentials:
    """
    Create Ernst potentials from Misra-Radhakrishna solution variables.

    The relation between metric forms:
    - MR: ds² = e^(λ-μ)(dt² - dρ²) - ρ²e^(-μ)dΦ² - e^μdz²
    - KJE: ds² = e^(2ψ)(dz - w dφ)² + ρ²e^(-2ψ)dφ² + e^(2(γ-ψ))(-dt² + dρ²)

    With w = 0 (no rotation):
    - e^μ = e^(2ψ)  →  ψ = μ/2

    For the electromagnetic part:
    - MR's ψ (potential) relates to A_z
    - MR's φ (potential) relates to χ

    Args:
        mu_func: Misra-Radhakrishna μ(ρ, t)
        psi_MR_func: Misra-Radhakrishna electromagnetic ψ(ρ, t)
        phi_func: Misra-Radhakrishna electromagnetic φ(ρ, t), default 0

    Returns:
        ErnstPotentials instance
    """
    def psi_func(rho, t):
        # e^μ_MR = e^(2ψ_Ernst) → ψ_Ernst = μ_MR / 2
        return mu_func(rho, t) / 2

    def A_z_func(rho, t):
        # The MR ψ potential is related to A_z
        # This is an approximation; exact relation depends on normalization
        return psi_MR_func(rho, t) / np.sqrt(8 * np.pi)

    def chi_func(rho, t):
        if phi_func is None:
            return 0.0
        return phi_func(rho, t) / np.sqrt(8 * np.pi)

    return ErnstPotentials(psi_func, None, A_z_func, chi_func)
