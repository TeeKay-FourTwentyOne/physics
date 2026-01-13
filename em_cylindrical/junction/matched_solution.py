"""
Matched Solution for Two-Region Spacetimes

Provides a unified interface for working with spacetimes that consist
of two regions separated by a junction hypersurface at ρ = R.

This is useful for:
- Matching interior solutions (with sources) to exterior solutions (vacuum)
- Modeling thin shells or domain walls
- Constructing composite spacetimes

The MatchedSolution automatically selects the appropriate region
based on the radial coordinate ρ.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

from .hypersurface import CylindricalHypersurface
from .junction_conditions import IsraelDarmoisConditions


class MatchedSolution:
    """
    Two-region solution with junction at ρ = R.

    Provides a seamless interface that automatically uses the interior
    solution for ρ < R and the exterior solution for ρ > R.

    At ρ = R exactly, uses the exterior solution by convention
    (approaching the junction from outside).
    """

    def __init__(self,
                 interior,
                 exterior,
                 junction_radius: float,
                 validate: bool = True,
                 tolerance: float = 1e-6):
        """
        Initialize a matched solution.

        Args:
            interior: Solution for ρ < R
            exterior: Solution for ρ > R
            junction_radius: The radius R where solutions are joined
            validate: Whether to validate junction conditions at creation
            tolerance: Tolerance for junction condition validation
        """
        if junction_radius <= 0:
            raise ValueError(f"Junction radius must be positive, got {junction_radius}")

        self.interior = interior
        self.exterior = exterior
        self.junction_radius = junction_radius
        self.R = junction_radius  # Alias
        self.tolerance = tolerance

        # Create junction condition checker
        self._junction = IsraelDarmoisConditions(
            interior, exterior, junction_radius, tolerance
        )

        # Optionally validate at creation
        if validate:
            self._validate_initial()

    def _validate_initial(self):
        """Check that first junction condition is approximately satisfied."""
        try:
            result = self._junction.first_condition(t=0.0)
            if not result['satisfied']:
                import warnings
                warnings.warn(
                    f"First junction condition not satisfied at t=0: "
                    f"max jump = {result['max_jump']:.2e}. "
                    f"Solutions may not match smoothly.",
                    RuntimeWarning
                )
        except Exception:
            # If we can't evaluate at t=0, skip validation
            pass

    def _select_solution(self, rho: float):
        """Select interior or exterior solution based on ρ."""
        if rho < self.R:
            return self.interior
        else:
            return self.exterior

    def psi(self, rho: float, t: float) -> float:
        """
        Electromagnetic potential ψ at (ρ, t).

        Args:
            rho: Radial coordinate
            t: Time coordinate

        Returns:
            Value of ψ
        """
        sol = self._select_solution(rho)
        return sol.psi(rho, t)

    def mu(self, rho: float, t: float) -> float:
        """
        Metric function μ at (ρ, t).

        Args:
            rho: Radial coordinate
            t: Time coordinate

        Returns:
            Value of μ
        """
        sol = self._select_solution(rho)
        return sol.mu(rho, t)

    def lambda_(self, rho: float, t: float) -> float:
        """
        Metric function λ at (ρ, t).

        Args:
            rho: Radial coordinate
            t: Time coordinate

        Returns:
            Value of λ
        """
        sol = self._select_solution(rho)
        return sol.lambda_(rho, t)

    def phi(self, rho: float, t: float) -> float:
        """
        Electromagnetic potential φ at (ρ, t).

        Args:
            rho: Radial coordinate
            t: Time coordinate

        Returns:
            Value of φ (0 for Case I, non-zero for Case II/III)
        """
        sol = self._select_solution(rho)
        if hasattr(sol, 'phi'):
            return sol.phi(rho, t)
        return 0.0

    def metric_at(self, rho: float, t: float) -> np.ndarray:
        """
        Compute metric tensor g_μν at (ρ, t).

        Args:
            rho: Radial coordinate
            t: Time coordinate

        Returns:
            4x4 metric tensor
        """
        sol = self._select_solution(rho)
        return sol.metric_at(rho, t)

    def region(self, rho: float) -> str:
        """
        Identify which region a point is in.

        Args:
            rho: Radial coordinate

        Returns:
            'interior' or 'exterior'
        """
        return 'interior' if rho < self.R else 'exterior'

    def is_at_junction(self, rho: float, tol: float = 1e-10) -> bool:
        """
        Check if a point is at the junction.

        Args:
            rho: Radial coordinate
            tol: Tolerance for comparison

        Returns:
            True if |ρ - R| < tol
        """
        return abs(rho - self.R) < tol

    def junction(self) -> IsraelDarmoisConditions:
        """
        Get the junction condition verifier.

        Returns:
            IsraelDarmoisConditions object for detailed analysis
        """
        return self._junction

    def hypersurface(self) -> CylindricalHypersurface:
        """
        Get the junction hypersurface.

        Returns:
            CylindricalHypersurface at ρ = R
        """
        return self._junction.hypersurface

    def is_valid(self, t: float) -> bool:
        """
        Check if junction conditions are satisfied at time t.

        This checks both the first condition (metric continuity)
        and identifies if a thin shell is required.

        Args:
            t: Time coordinate

        Returns:
            True if first junction condition is satisfied
        """
        result = self._junction.first_condition(t)
        return result['satisfied']

    def has_thin_shell(self, t: float) -> bool:
        """
        Check if a thin shell is present at the junction.

        Args:
            t: Time coordinate

        Returns:
            True if [K_ab] ≠ 0 (thin shell required)
        """
        result = self._junction.second_condition(t)
        return result['thin_shell']

    def shell_properties(self, t: float) -> Optional[Dict[str, float]]:
        """
        Get properties of the thin shell at the junction.

        Args:
            t: Time coordinate

        Returns:
            Dictionary with shell properties, or None if no shell
        """
        if not self.has_thin_shell(t):
            return None

        sigma = self._junction.surface_energy_density(t)
        p_Phi, p_z = self._junction.surface_pressures(t)

        return {
            'energy_density': sigma,
            'pressure_Phi': p_Phi,
            'pressure_z': p_z,
            'equation_of_state_Phi': p_Phi / sigma if abs(sigma) > 1e-15 else float('inf'),
            'equation_of_state_z': p_z / sigma if abs(sigma) > 1e-15 else float('inf'),
            'weak_energy': self._junction.weak_energy_condition(t),
            'dominant_energy': self._junction.dominant_energy_condition(t),
        }

    def discontinuity_at_junction(self, t: float) -> Dict[str, float]:
        """
        Measure the discontinuity in physical quantities at the junction.

        Args:
            t: Time coordinate

        Returns:
            Dictionary with jumps in various quantities
        """
        R = self.R

        # Metric function jumps
        mu_int = self.interior.mu(R, t)
        mu_ext = self.exterior.mu(R, t)
        lam_int = self.interior.lambda_(R, t)
        lam_ext = self.exterior.lambda_(R, t)

        # EM potential jumps
        psi_int = self.interior.psi(R, t)
        psi_ext = self.exterior.psi(R, t)

        phi_int = self.interior.phi(R, t) if hasattr(self.interior, 'phi') else 0
        phi_ext = self.exterior.phi(R, t) if hasattr(self.exterior, 'phi') else 0

        return {
            'mu_jump': mu_ext - mu_int,
            'lambda_jump': lam_ext - lam_int,
            'psi_jump': psi_ext - psi_int,
            'phi_jump': phi_ext - phi_int,
        }

    def evaluate_profile(self, t: float, rho_min: float = 0.1,
                        rho_max: float = None, n_points: int = 100
                        ) -> Dict[str, np.ndarray]:
        """
        Evaluate solution along a radial profile at fixed time.

        Useful for visualization and analysis.

        Args:
            t: Time coordinate
            rho_min: Minimum radius (must be > 0)
            rho_max: Maximum radius (default: 3 * junction_radius)
            n_points: Number of sample points

        Returns:
            Dictionary with arrays for rho, mu, lambda, psi, phi, region
        """
        if rho_max is None:
            rho_max = 3 * self.R

        rho_values = np.linspace(rho_min, rho_max, n_points)

        mu_values = np.array([self.mu(r, t) for r in rho_values])
        lam_values = np.array([self.lambda_(r, t) for r in rho_values])
        psi_values = np.array([self.psi(r, t) for r in rho_values])
        phi_values = np.array([self.phi(r, t) for r in rho_values])
        regions = np.array([self.region(r) for r in rho_values])

        return {
            'rho': rho_values,
            'mu': mu_values,
            'lambda': lam_values,
            'psi': psi_values,
            'phi': phi_values,
            'region': regions,
            'junction_radius': self.R,
        }

    def summary(self, t: float = 0.0) -> str:
        """
        Generate a summary of the matched solution.

        Args:
            t: Time for evaluation

        Returns:
            Formatted summary string
        """
        lines = [
            "Matched Solution Summary",
            "=" * 60,
            f"Junction radius: R = {self.R}",
            f"Interior solution: {type(self.interior).__name__}",
            f"Exterior solution: {type(self.exterior).__name__}",
            "",
            f"Evaluation at t = {t}:",
        ]

        # Check junction conditions
        valid = self.is_valid(t)
        has_shell = self.has_thin_shell(t)

        lines.extend([
            f"  First condition satisfied: {valid}",
            f"  Thin shell present: {has_shell}",
        ])

        # Discontinuity info
        disc = self.discontinuity_at_junction(t)
        lines.extend([
            "",
            "Discontinuities at junction:",
            f"  [μ] = {disc['mu_jump']:.6e}",
            f"  [λ] = {disc['lambda_jump']:.6e}",
            f"  [ψ] = {disc['psi_jump']:.6e}",
            f"  [φ] = {disc['phi_jump']:.6e}",
        ])

        # Shell properties
        if has_shell:
            shell = self.shell_properties(t)
            lines.extend([
                "",
                "Thin Shell Properties:",
                f"  Energy density σ = {shell['energy_density']:.6e}",
                f"  Pressure p_Φ = {shell['pressure_Phi']:.6e}",
                f"  Pressure p_z = {shell['pressure_z']:.6e}",
                f"  WEC satisfied: {shell['weak_energy']}",
                f"  DEC satisfied: {shell['dominant_energy']}",
            ])

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"MatchedSolution(R={self.R}, "
                f"interior={type(self.interior).__name__}, "
                f"exterior={type(self.exterior).__name__})")
