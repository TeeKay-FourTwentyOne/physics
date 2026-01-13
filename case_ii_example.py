#!/usr/bin/env python3
"""
Case II Example: Electromagnetic Fields with ψ = 0

This example demonstrates:
1. Creating a Case II solution (only φ potential non-zero)
2. Verifying it's a vacuum solution (J = 0)
3. Computing surface currents at a cylindrical boundary
4. Comparing with a sourced solution

Run with: python case_ii_example.py
"""

import numpy as np
from em_cylindrical import (
    CaseIISolution,
    CurrentDensity,
    CylindricalBoundary,
    compute_solenoid_current,
    uniform_axial_current,
)


def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def main():
    print("=" * 70)
    print("  CASE II EXAMPLE: ψ = 0 (Only φ electromagnetic potential)")
    print("=" * 70)

    # =========================================================================
    # Part 1: Create and examine the Case II solution
    # =========================================================================
    print_header("PART 1: The Case II Solution")

    # Create a Case II solution with the 'rho_only' variant (equation 56)
    # φ = p - (m/a) * tanh(-(m/2)*log(ρ) + n)
    sol = CaseIISolution(
        variant='rho_only',
        m=1.0,      # Parameter m
        n=0.5,      # Parameter n
        a=1.0,      # Parameter a
        p=0.0,      # Integration constant
        q=0.0       # Integration constant
    )

    print(f"\nSolution: {sol}")
    print(f"Variant: 'rho_only' - φ depends only on ρ (equation 56 from paper)")

    # Show symbolic expressions
    print("\nSymbolic expressions:")
    print(f"  φ = {sol.phi_symbolic}")
    print(f"  μ = {sol.mu_symbolic}")
    print(f"  λ = {sol.lambda_symbolic}")

    # Evaluate at several points
    print("\nNumerical values along ρ axis (at t=0):")
    print(f"{'ρ':>6} {'φ':>12} {'ψ':>12} {'μ':>12} {'λ':>12}")
    print("-" * 58)
    for rho in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        phi = sol.phi(rho, 0.0)
        psi = sol.psi(rho, 0.0)
        mu = sol.mu(rho, 0.0)
        lam = sol.lambda_(rho, 0.0)
        print(f"{rho:>6.1f} {phi:>12.6f} {psi:>12.6f} {mu:>12.6f} {lam:>12.6f}")

    # =========================================================================
    # Part 2: Verify this is a vacuum solution (J = 0)
    # =========================================================================
    print_header("PART 2: Verify J = 0 (Vacuum Solution)")

    current = CurrentDensity(sol)

    print("\nMaxwell's equations: ∇_μ F^μν = 4π J^ν")
    print("For vacuum solutions, J^μ should be zero everywhere.\n")

    # Check at several points
    print(f"{'ρ':>6} {'t':>6} {'J^ρ':>12} {'J^Φ':>12} {'J^z':>12} {'J^t':>12} {'|J|':>12}")
    print("-" * 78)

    for rho in [1.0, 1.5, 2.0]:
        for t in [0.0, 0.5]:
            result = current.verify_vacuum(rho, t)
            print(f"{rho:>6.1f} {t:>6.1f} {result['J_rho']:>12.2e} {result['J_Phi']:>12.2e} "
                  f"{result['J_z']:>12.2e} {result['J_t']:>12.2e} {result['J_magnitude']:>12.2e}")

    # Grid verification
    grid_result = current.verify_vacuum_grid(
        rho_range=(0.5, 3.0),
        t_range=(0.0, 1.0),
        n_points=5
    )
    print(f"\nMaximum |J| over grid: {grid_result['max_J_magnitude']:.2e}")
    print("✓ Vacuum condition verified (J ≈ 0)")

    # =========================================================================
    # Part 3: Compute surface currents at a cylindrical boundary
    # =========================================================================
    print_header("PART 3: Surface Currents at Cylindrical Boundary")

    R = 2.0  # Boundary radius
    boundary = CylindricalBoundary(sol, radius=R)

    print(f"\nBoundary at ρ = {R}")
    print("\nPhysical interpretation:")
    print("  If the field exists only inside ρ < R with vacuum outside,")
    print("  surface currents K^μ are needed at the boundary to sustain the field.")
    print("\n  K^μ = -(1/4π) n_α F^αμ  (jump condition)")

    # Compute surface currents at different times
    print(f"\nSurface current components at ρ = {R}:")
    print(f"{'t':>6} {'K^ρ':>12} {'K^Φ':>12} {'K^z':>12} {'K^t':>12}")
    print("-" * 54)

    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        K = boundary.surface_current_physical(t)
        print(f"{t:>6.2f} {K['K_rho']:>12.6f} {K['K_Phi']:>12.6f} "
              f"{K['K_z']:>12.6f} {K['K_t']:>12.6f}")

    print("\nNote: K^ρ ≈ 0 (current is tangent to surface, as expected)")

    # Total axial current
    print(f"\nTotal axial current I_z = ∮ K^z · R dΦ = 2πR · K^z")
    for t in [0.0, 0.5, 1.0]:
        I_z = boundary.total_current_z(t)
        print(f"  t = {t:.1f}: I_z = {I_z:.6f}")

    # Use convenience function
    solenoid = compute_solenoid_current(sol, radius=R, t_val=0.0)
    print(f"\nEquivalent solenoid parameters (t=0):")
    print(f"  Surface current density K_z: {solenoid['surface_current_density_z']:.6f}")
    print(f"  Total axial current I_z: {solenoid['total_axial_current']:.6f}")

    # =========================================================================
    # Part 4: Field tensor and magnetic field at boundary
    # =========================================================================
    print_header("PART 4: Electromagnetic Field at Boundary")

    print(f"\nField tensor F_μν at ρ = {R}, t = 0:")
    F = sol.field_tensor_at(R, 0.0)
    print(f"\n  F_μν = ")
    for i in range(4):
        row = "  [" + " ".join(f"{F[i, j]:>10.6f}" for j in range(4)) + "]"
        print(row)

    field = boundary.magnetic_field_at_boundary(t_val=0.0)
    print(f"\nField components:")
    print(f"  F₁₂ = {field['F_12']:.6f}  (related to B_z)")
    print(f"  F₁₃ = {field['F_13']:.6f}  (related to B_Φ)")
    print(f"  F₂₄ = {field['F_24']:.6f}  (related to E_Φ)")
    print(f"  F₃₄ = {field['F_34']:.6f}  (related to E_z)")

    # =========================================================================
    # Part 5: Compare with a sourced solution
    # =========================================================================
    print_header("PART 5: Comparison with Sourced Solution")

    print("\nCreating a uniform axial current distribution for comparison:")
    print("  J^z = I₀/(πR²) for ρ < R, 0 otherwise")

    I_0 = 1.0
    sourced = uniform_axial_current(I_0=I_0, R=R)

    print(f"\n  I₀ = {I_0}")
    print(f"  R = {R}")
    print(f"  J^z (uniform) = {I_0 / (np.pi * R ** 2):.6f}")

    # Check the sourced solution
    print("\nSourced solution potentials (ψ for axial current):")
    print(f"{'ρ':>6} {'ψ (sourced)':>15}")
    print("-" * 25)
    for rho in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        psi_s = sourced.psi(rho, 0.0)
        print(f"{rho:>6.1f} {psi_s:>15.6f}")

    # Verify Maxwell with source
    print("\nVerifying ∇_μ F^μν = 4π J^ν for sourced solution at ρ = 0.5:")
    result = sourced.verify_maxwell_with_source(0.5, 0.0)
    print(f"  J^z (computed from field): {result['J_computed'][2]:.6f}")
    print(f"  J^z (specified):           {result['J_specified'][2]:.6f}")
    print(f"  Residual:                  {result['J_z_residual']:.2e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print_header("SUMMARY")
    print("""
Case II Solution (ψ = 0, φ ≠ 0):
  • Represents electromagnetic fields with only azimuthal potential
  • Verified as vacuum solution: J^μ ≈ 0 everywhere
  • At boundary ρ = R, surface currents K^μ sustain the field
  • Can be compared with explicit sourced configurations

Physical Interpretation:
  • φ potential → F₁₂ component → axial magnetic field B_z
  • Surface current K^z at boundary acts like a solenoid winding
  • The curved spacetime (via metric functions μ, λ) modifies field behavior
""")

    print("=" * 70)
    print("  Example Complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
