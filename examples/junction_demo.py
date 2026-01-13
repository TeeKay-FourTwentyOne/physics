#!/usr/bin/env python3
"""
Israel-Darmois Junction Conditions Demo

Demonstrates the junction conditions module for matching spacetime regions
across cylindrical hypersurfaces (ρ = R).

This module implements the Israel-Darmois formalism:
1. First condition (Darmois): [γ_ab] = 0 (induced metric continuity)
2. Second condition (Israel): [K_ab] = -8π(S_ab - ½γ_ab S)
   (extrinsic curvature jump relates to surface stress-energy)

Key features demonstrated:
1. Hypersurface geometry (induced metric, normal vectors)
2. Extrinsic curvature calculation (K_ab)
3. Junction condition verification
4. Thin shell analysis (surface stress-energy)
5. Matched two-region solutions
"""

import numpy as np

from em_cylindrical import CaseISolution, CaseIISolution
from em_cylindrical.junction import (
    CylindricalHypersurface,
    ExtrinsicCurvature,
    IsraelDarmoisConditions,
    MatchedSolution,
)


def demo_hypersurface_geometry():
    """Demonstrate hypersurface geometry calculations."""
    print("=" * 60)
    print("Demo 1: Hypersurface Geometry")
    print("=" * 60)
    print("\nFor a ρ = R hypersurface in Einstein-Rosen spacetime,")
    print("the intrinsic geometry is described by the induced metric γ_ab.")

    # Create a solution and hypersurface
    sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
    R = 2.0
    t = 1.0

    hs = CylindricalHypersurface(radius=R)

    print(f"\nHypersurface at R = {R}")
    print(f"Tangent coordinates: {hs.tangent_coordinates()}")

    # Compute induced metric
    gamma = hs.induced_metric(sol, t)
    print(f"\nInduced metric γ_ab at t = {t}:")
    print(f"  γ_ΦΦ = {gamma[0, 0]:.6f}")
    print(f"  γ_zz = {gamma[1, 1]:.6f}")
    print(f"  γ_tt = {gamma[2, 2]:.6f}")

    # Check signature (should be (-,-,+) for timelike hypersurface)
    signs = np.sign(np.diag(gamma))
    print(f"\nSignature: ({int(signs[0]):+d}, {int(signs[1]):+d}, {int(signs[2]):+d})")
    print("(Should be (-, -, +) for timelike hypersurface)")

    # Normal vector
    n = hs.normal_vector(sol, t)
    n_cov = hs.normal_covector(sol, t)
    print(f"\nNormal vector n^μ = ({n[0]:.6f}, {n[1]:.6f}, {n[2]:.6f}, {n[3]:.6f})")
    print(f"Normal covector n_μ = ({n_cov[0]:.6f}, {n_cov[1]:.6f}, {n_cov[2]:.6f}, {n_cov[3]:.6f})")

    # Area element
    area = hs.area_element(sol, t)
    det = hs.induced_metric_determinant(sol, t)
    print(f"\nArea element √|det(γ)| = {area:.6f}")
    print(f"Determinant det(γ) = {det:.6f}")


def demo_extrinsic_curvature():
    """Demonstrate extrinsic curvature calculations."""
    print("\n" + "=" * 60)
    print("Demo 2: Extrinsic Curvature (Second Fundamental Form)")
    print("=" * 60)
    print("\nThe extrinsic curvature K_ab measures how the hypersurface")
    print("bends within the ambient 4D spacetime.")

    sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
    hs = CylindricalHypersurface(radius=2.0)
    t = 1.0

    K = ExtrinsicCurvature(hs, sol)

    # Compute K_ab
    K_ab = K.K_ab(t)
    print(f"\nExtrinsic curvature K_ab at t = {t}:")
    print(f"  K_ΦΦ = {K_ab[0, 0]:.6f}")
    print(f"  K_zz = {K_ab[1, 1]:.6f}")
    print(f"  K_tt = {K_ab[2, 2]:.6f}")

    # Trace (mean curvature)
    trace = K.K_trace(t)
    print(f"\nMean curvature K = γ^ab K_ab = {trace:.6f}")

    # Principal curvatures
    principal = K.principal_curvatures(t)
    print(f"\nPrincipal curvatures:")
    for i, p in enumerate(principal):
        print(f"  κ_{i+1} = {p:.6f}")

    # Gaussian curvature contribution
    gauss = K.gaussian_curvature_contribution(t)
    print(f"\nGauss contribution K² - K_ab K^ab = {gauss:.6f}")


def demo_junction_conditions():
    """Demonstrate Israel-Darmois junction conditions."""
    print("\n" + "=" * 60)
    print("Demo 3: Israel-Darmois Junction Conditions")
    print("=" * 60)
    print("\nThe junction conditions determine whether two spacetime")
    print("regions can be smoothly joined across a hypersurface.")

    # Same solution on both sides (should match perfectly)
    print("\n--- Case A: Identical Solutions ---")
    sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

    jc = IsraelDarmoisConditions(sol, sol, radius=2.0)

    result = jc.verify_all(t=1.0)

    print(f"\nFirst condition (metric continuity):")
    print(f"  Satisfied: {result['first_condition']['satisfied']}")
    print(f"  Max jump: {result['first_condition']['max_jump']:.2e}")

    print(f"\nSecond condition (extrinsic curvature):")
    print(f"  Thin shell required: {result['second_condition']['thin_shell']}")
    print(f"  Max K jump: {result['second_condition']['max_jump']:.2e}")

    # Different solutions (may require thin shell)
    print("\n--- Case B: Different Solutions ---")
    sol_int = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
    sol_ext = CaseISolution(alpha=1.2, beta=0.6, variant='t_only', m=1.2)

    jc2 = IsraelDarmoisConditions(sol_int, sol_ext, radius=2.0, tolerance=0.01)

    result2 = jc2.verify_all(t=1.0)

    print(f"\nFirst condition (metric continuity):")
    print(f"  Satisfied: {result2['first_condition']['satisfied']}")
    jumps = result2['first_condition']['component_jumps']
    print(f"  [γ_ΦΦ] = {jumps['gamma_PhiPhi']:.6f}")
    print(f"  [γ_zz] = {jumps['gamma_zz']:.6f}")
    print(f"  [γ_tt] = {jumps['gamma_tt']:.6f}")

    print(f"\nSecond condition (extrinsic curvature):")
    print(f"  Thin shell required: {result2['second_condition']['thin_shell']}")

    if result2['shell'] is not None:
        shell = result2['shell']
        print(f"\nThin Shell Properties:")
        print(f"  Energy density σ = {shell['energy_density']:.6f}")
        print(f"  Pressure p_Φ = {shell['pressure_Phi']:.6f}")
        print(f"  Pressure p_z = {shell['pressure_z']:.6f}")
        print(f"  Weak Energy Condition: {shell['weak_energy_condition']}")
        print(f"  Dominant Energy Condition: {shell['dominant_energy_condition']}")


def demo_matched_solution():
    """Demonstrate MatchedSolution for two-region spacetimes."""
    print("\n" + "=" * 60)
    print("Demo 4: Matched Two-Region Solution")
    print("=" * 60)
    print("\nMatchedSolution provides a unified interface for spacetimes")
    print("consisting of interior (ρ < R) and exterior (ρ > R) regions.")

    # Create matched solution with same solutions (smooth matching)
    sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

    matched = MatchedSolution(sol, sol, junction_radius=2.0, validate=False)

    print(f"\nJunction radius: R = {matched.R}")

    # Evaluate in different regions
    t = 1.0
    points = [1.0, 1.5, 2.0, 2.5, 3.0]

    print(f"\nMetric function μ along radial direction (t = {t}):")
    print("-" * 40)
    print(f"{'ρ':>8} {'Region':>10} {'μ':>12}")
    print("-" * 40)
    for rho in points:
        mu = matched.mu(rho, t)
        region = matched.region(rho)
        marker = "*" if matched.is_at_junction(rho) else " "
        print(f"{rho:>8.2f} {region:>10} {mu:>12.6f} {marker}")
    print("-" * 40)
    print("* = at junction")

    # Junction condition status
    print(f"\nJunction Status:")
    print(f"  Valid junction: {matched.is_valid(t)}")
    print(f"  Has thin shell: {matched.has_thin_shell(t)}")

    # Discontinuity analysis
    disc = matched.discontinuity_at_junction(t)
    print(f"\nDiscontinuities at junction:")
    print(f"  [μ] = {disc['mu_jump']:.2e}")
    print(f"  [λ] = {disc['lambda_jump']:.2e}")
    print(f"  [ψ] = {disc['psi_jump']:.2e}")


def demo_case_ii_junction():
    """Demonstrate junction conditions with Case II solution."""
    print("\n" + "=" * 60)
    print("Demo 5: Case II Solution Junction")
    print("=" * 60)

    # Create Case II solution
    sol = CaseIISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

    hs = CylindricalHypersurface(radius=2.0)
    t = 0.5

    print(f"\nCase II Solution (sinh variant) at R = {hs.radius}, t = {t}")

    # Induced metric
    gamma = hs.induced_metric(sol, t)
    print(f"\nInduced metric:")
    print(f"  γ_ΦΦ = {gamma[0, 0]:.6f}")
    print(f"  γ_zz = {gamma[1, 1]:.6f}")
    print(f"  γ_tt = {gamma[2, 2]:.6f}")

    # Extrinsic curvature
    K = ExtrinsicCurvature(hs, sol)
    K_ab = K.K_ab(t)
    print(f"\nExtrinsic curvature:")
    print(f"  K_ΦΦ = {K_ab[0, 0]:.6f}")
    print(f"  K_zz = {K_ab[1, 1]:.6f}")
    print(f"  K_tt = {K_ab[2, 2]:.6f}")
    print(f"  K (trace) = {K.K_trace(t):.6f}")


def demo_physical_interpretation():
    """Discuss physical interpretation of junction conditions."""
    print("\n" + "=" * 60)
    print("Demo 6: Physical Interpretation")
    print("=" * 60)

    print("""
Physical meaning of junction conditions:

1. FIRST CONDITION [γ_ab] = 0:
   - The intrinsic geometry of the hypersurface must be
     the same when viewed from either side.
   - This ensures there's no "gap" or "overlap" in the
     spacetime manifold.
   - Physically: observers on the shell agree on distances
     and time intervals.

2. SECOND CONDITION [K_ab] = -8π(S_ab - ½γ_ab S):
   - If K_ab is continuous ([K_ab] = 0), the junction is smooth
     with no matter on the shell.
   - If K_ab has a jump, there must be a thin shell of matter
     with surface stress-energy S_ab.

Physical applications:
   - Matching interior stellar solutions to exterior vacuum
   - Modeling domain walls in cosmology
   - Describing shells of matter around compact objects
   - Constructing gravitational wave sources

For cylindrical symmetry:
   - The shell is a cylindrical surface at ρ = R
   - Surface energy density σ = S^t_t
   - Surface pressures p_Φ and p_z describe tangential stresses
""")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 60)
    print("# Israel-Darmois Junction Conditions Demo")
    print("#" * 60)
    print("\nBased on:")
    print("- Israel (1966), Nuovo Cimento B 44, 1")
    print("- Darmois (1927), Mémorial des Sciences Mathématiques")

    demo_hypersurface_geometry()
    demo_extrinsic_curvature()
    demo_junction_conditions()
    demo_matched_solution()
    demo_case_ii_junction()
    demo_physical_interpretation()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
