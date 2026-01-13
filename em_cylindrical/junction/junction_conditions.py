"""
Israel-Darmois Junction Conditions

Implements verification of the Israel-Darmois junction conditions for
matching spacetime regions across cylindrical hypersurfaces.

The junction conditions are:
1. First condition (Darmois): [γ_ab] = 0
   The induced metric must be continuous across the hypersurface.

2. Second condition (Israel): [K_ab] = -8π(S_ab - ½γ_ab S)
   The jump in extrinsic curvature relates to surface stress-energy.

For vacuum matching (no thin shell): S_ab = 0, so [K_ab] = 0.

References:
    - Israel (1966), Nuovo Cimento B 44, 1
    - Darmois (1927), Mémorial des Sciences Mathématiques
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional

from .hypersurface import CylindricalHypersurface
from .extrinsic_curvature import ExtrinsicCurvature


class IsraelDarmoisConditions:
    """
    Verifies Israel-Darmois junction conditions for two spacetime regions.

    Given interior and exterior solutions separated by a hypersurface at
    ρ = R, this class checks whether the junction conditions are satisfied
    and computes any surface stress-energy that may be present.
    """

    # Factor relating geometry to stress-energy (8π in geometric units)
    KAPPA = 8 * np.pi

    def __init__(self,
                 solution_interior,
                 solution_exterior,
                 radius: float,
                 tolerance: float = 1e-6):
        """
        Initialize junction condition checker.

        Args:
            solution_interior: Solution for ρ < R (approaching from inside)
            solution_exterior: Solution for ρ > R (approaching from outside)
            radius: Junction radius R where hypersurface is located
            tolerance: Tolerance for considering jumps as zero
        """
        if radius <= 0:
            raise ValueError(f"Radius must be positive, got {radius}")

        self.interior = solution_interior
        self.exterior = solution_exterior
        self.radius = radius
        self.tolerance = tolerance

        # Create hypersurface and curvature calculators
        self.hypersurface = CylindricalHypersurface(radius)
        self.K_interior = ExtrinsicCurvature(self.hypersurface, solution_interior)
        self.K_exterior = ExtrinsicCurvature(self.hypersurface, solution_exterior)

    def induced_metric_interior(self, t: float) -> np.ndarray:
        """Get induced metric from interior solution."""
        return self.hypersurface.induced_metric(self.interior, t)

    def induced_metric_exterior(self, t: float) -> np.ndarray:
        """Get induced metric from exterior solution."""
        return self.hypersurface.induced_metric(self.exterior, t)

    def induced_metric_jump(self, t: float) -> np.ndarray:
        """
        Compute the jump in induced metric: [γ_ab] = γ_ab⁺ - γ_ab⁻

        Args:
            t: Time coordinate

        Returns:
            3x3 matrix of metric jumps
        """
        gamma_plus = self.induced_metric_exterior(t)
        gamma_minus = self.induced_metric_interior(t)
        return gamma_plus - gamma_minus

    def extrinsic_curvature_interior(self, t: float) -> np.ndarray:
        """Get extrinsic curvature from interior solution."""
        return self.K_interior.K_ab(t)

    def extrinsic_curvature_exterior(self, t: float) -> np.ndarray:
        """Get extrinsic curvature from exterior solution."""
        return self.K_exterior.K_ab(t)

    def extrinsic_curvature_jump(self, t: float) -> np.ndarray:
        """
        Compute the jump in extrinsic curvature: [K_ab] = K_ab⁺ - K_ab⁻

        Args:
            t: Time coordinate

        Returns:
            3x3 matrix of curvature jumps
        """
        K_plus = self.extrinsic_curvature_exterior(t)
        K_minus = self.extrinsic_curvature_interior(t)
        return K_plus - K_minus

    def first_condition(self, t: float) -> Dict[str, Any]:
        """
        Check the first junction condition (Darmois): [γ_ab] = 0

        The induced metric must be continuous across the hypersurface
        for a valid junction.

        Args:
            t: Time coordinate

        Returns:
            Dictionary with:
            - 'gamma_jump': 3x3 matrix of metric jumps
            - 'max_jump': Maximum absolute jump value
            - 'satisfied': Whether condition is satisfied within tolerance
            - 'component_jumps': Individual component jumps
        """
        gamma_jump = self.induced_metric_jump(t)
        max_jump = np.max(np.abs(gamma_jump))

        return {
            'gamma_jump': gamma_jump,
            'max_jump': max_jump,
            'satisfied': max_jump < self.tolerance,
            'component_jumps': {
                'gamma_PhiPhi': gamma_jump[0, 0],
                'gamma_zz': gamma_jump[1, 1],
                'gamma_tt': gamma_jump[2, 2],
            }
        }

    def second_condition(self, t: float) -> Dict[str, Any]:
        """
        Check the second junction condition (Israel):
        [K_ab] = -κ(S_ab - ½γ_ab S)

        Where κ = 8π, S_ab is surface stress-energy, and S = γ^ab S_ab.

        If [K_ab] = 0, we have vacuum matching (no thin shell).
        If [K_ab] ≠ 0, a thin shell with surface stress-energy exists.

        Args:
            t: Time coordinate

        Returns:
            Dictionary with:
            - 'K_jump': 3x3 matrix of curvature jumps
            - 'K_trace_jump': Jump in trace of K
            - 'max_jump': Maximum absolute jump value
            - 'thin_shell': Whether a thin shell is required
            - 'satisfied': Whether condition can be satisfied
        """
        K_jump = self.extrinsic_curvature_jump(t)
        max_jump = np.max(np.abs(K_jump))

        # Compute trace jump
        gamma_inv = self.hypersurface.induced_metric_inverse(self.exterior, t)
        K_trace_jump = np.sum(gamma_inv * K_jump)

        thin_shell = max_jump >= self.tolerance

        return {
            'K_jump': K_jump,
            'K_trace_jump': K_trace_jump,
            'max_jump': max_jump,
            'thin_shell': thin_shell,
            'satisfied': True,  # Always satisfiable with appropriate S_ab
            'component_jumps': {
                'K_PhiPhi': K_jump[0, 0],
                'K_zz': K_jump[1, 1],
                'K_tt': K_jump[2, 2],
            }
        }

    def surface_stress_energy(self, t: float) -> np.ndarray:
        """
        Compute surface stress-energy tensor S_ab from [K_ab].

        From the Israel junction condition:
        [K_ab] = -κ(S_ab - ½γ_ab S)

        Solving for S_ab:
        S_ab = -(1/κ)([K_ab] - γ_ab [K])

        Where [K] = γ^ab [K_ab] is the trace of the curvature jump.

        Args:
            t: Time coordinate

        Returns:
            3x3 surface stress-energy tensor S_ab
        """
        K_jump = self.extrinsic_curvature_jump(t)
        gamma = self.hypersurface.induced_metric(self.exterior, t)
        gamma_inv = self.hypersurface.induced_metric_inverse(self.exterior, t)

        # Trace of curvature jump
        K_trace_jump = np.sum(gamma_inv * K_jump)

        # S_ab = -(1/κ)([K_ab] - γ_ab [K])
        S_ab = -(1.0 / self.KAPPA) * (K_jump - gamma * K_trace_jump)

        return S_ab

    def surface_stress_energy_mixed(self, t: float) -> np.ndarray:
        """
        Compute mixed surface stress-energy tensor S^a_b.

        S^a_b = γ^ac S_cb

        This form is useful for identifying physical components.

        Args:
            t: Time coordinate

        Returns:
            3x3 mixed tensor S^a_b
        """
        S_ab = self.surface_stress_energy(t)
        gamma_inv = self.hypersurface.induced_metric_inverse(self.exterior, t)

        return gamma_inv @ S_ab

    def surface_energy_density(self, t: float) -> float:
        """
        Compute surface energy density σ.

        σ = S^t_t (the time-time component of mixed stress-energy)

        This represents the energy per unit area on the shell.

        Args:
            t: Time coordinate

        Returns:
            Surface energy density
        """
        S_mixed = self.surface_stress_energy_mixed(t)
        # t is the third index (0=Φ, 1=z, 2=t)
        return S_mixed[2, 2]

    def surface_pressures(self, t: float) -> Tuple[float, float]:
        """
        Compute surface pressures (p_Φ, p_z).

        p_Φ = -S^Φ_Φ
        p_z = -S^z_z

        These represent the tangential pressures on the shell.

        Args:
            t: Time coordinate

        Returns:
            Tuple (p_Φ, p_z) of surface pressures
        """
        S_mixed = self.surface_stress_energy_mixed(t)

        p_Phi = -S_mixed[0, 0]
        p_z = -S_mixed[1, 1]

        return p_Phi, p_z

    def weak_energy_condition(self, t: float) -> bool:
        """
        Check if weak energy condition is satisfied on the shell.

        WEC requires: σ ≥ 0 (positive energy density)

        Args:
            t: Time coordinate

        Returns:
            True if WEC is satisfied
        """
        sigma = self.surface_energy_density(t)
        return sigma >= -self.tolerance

    def dominant_energy_condition(self, t: float) -> bool:
        """
        Check if dominant energy condition is satisfied on the shell.

        DEC requires: σ ≥ 0 and σ ≥ |p_i| for all pressures

        Args:
            t: Time coordinate

        Returns:
            True if DEC is satisfied
        """
        sigma = self.surface_energy_density(t)
        p_Phi, p_z = self.surface_pressures(t)

        return (sigma >= -self.tolerance and
                sigma >= abs(p_Phi) - self.tolerance and
                sigma >= abs(p_z) - self.tolerance)

    def verify_all(self, t: float) -> Dict[str, Any]:
        """
        Complete verification of all junction conditions.

        Args:
            t: Time coordinate

        Returns:
            Comprehensive dictionary with all junction condition results
        """
        first = self.first_condition(t)
        second = self.second_condition(t)

        result = {
            'time': t,
            'radius': self.radius,
            'first_condition': first,
            'second_condition': second,
            'valid_junction': first['satisfied'],
        }

        # Add shell properties if thin shell exists
        if second['thin_shell']:
            sigma = self.surface_energy_density(t)
            p_Phi, p_z = self.surface_pressures(t)

            result['shell'] = {
                'energy_density': sigma,
                'pressure_Phi': p_Phi,
                'pressure_z': p_z,
                'weak_energy_condition': self.weak_energy_condition(t),
                'dominant_energy_condition': self.dominant_energy_condition(t),
            }
        else:
            result['shell'] = None

        return result

    def summary(self, t: float) -> str:
        """
        Generate a human-readable summary of junction conditions.

        Args:
            t: Time coordinate

        Returns:
            Formatted string summary
        """
        result = self.verify_all(t)

        lines = [
            f"Israel-Darmois Junction Conditions at ρ = {self.radius}, t = {t}",
            "=" * 60,
            "",
            "First Condition (Induced Metric Continuity):",
            f"  [γ_ΦΦ] = {result['first_condition']['component_jumps']['gamma_PhiPhi']:.6e}",
            f"  [γ_zz] = {result['first_condition']['component_jumps']['gamma_zz']:.6e}",
            f"  [γ_tt] = {result['first_condition']['component_jumps']['gamma_tt']:.6e}",
            f"  Satisfied: {result['first_condition']['satisfied']}",
            "",
            "Second Condition (Extrinsic Curvature):",
            f"  [K_ΦΦ] = {result['second_condition']['component_jumps']['K_PhiPhi']:.6e}",
            f"  [K_zz] = {result['second_condition']['component_jumps']['K_zz']:.6e}",
            f"  [K_tt] = {result['second_condition']['component_jumps']['K_tt']:.6e}",
            f"  Thin Shell Required: {result['second_condition']['thin_shell']}",
        ]

        if result['shell'] is not None:
            shell = result['shell']
            lines.extend([
                "",
                "Thin Shell Properties:",
                f"  Energy Density σ = {shell['energy_density']:.6e}",
                f"  Pressure p_Φ = {shell['pressure_Phi']:.6e}",
                f"  Pressure p_z = {shell['pressure_z']:.6e}",
                f"  Weak Energy Condition: {shell['weak_energy_condition']}",
                f"  Dominant Energy Condition: {shell['dominant_energy_condition']}",
            ])

        lines.extend([
            "",
            f"Valid Junction: {result['valid_junction']}",
        ])

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"IsraelDarmoisConditions(radius={self.radius}, "
                f"interior={type(self.interior).__name__}, "
                f"exterior={type(self.exterior).__name__})")
