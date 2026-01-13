#!/usr/bin/env python3
"""
Demonstration of Einstein-Maxwell Cylindrical Symmetry Solutions

This script demonstrates the use of exact solutions from Misra & Radhakrishna (1962)
for electromagnetic fields with cylindrical symmetry in general relativity.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from em_cylindrical import CaseISolution, CaseIISolution, CaseIIISolution
from em_cylindrical.metric import EinsteinRosenMetric


def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def demo_case_i():
    """Demonstrate Case (i): φ = 0 solutions."""
    print_section("Case (i): phi = 0 (Only psi non-zero)")

    # Create solution with t_only variant (equation 43)
    sol = CaseISolution(
        alpha=1.0,
        beta=0.5,
        variant='t_only',
        m=2.0,
        n=0.0,
        p=1.0,
        q=0.0
    )
    print(f"\nSolution: {sol}")

    # Evaluate at a point
    rho_val, t_val = 2.0, 1.0
    print(f"\nEvaluating at (rho={rho_val}, t={t_val}):")

    print(f"  phi = {sol.phi(rho_val, t_val):.6f} (always 0 for Case i)")
    print(f"  psi = {sol.psi(rho_val, t_val):.6f}")
    print(f"  mu  = {sol.mu(rho_val, t_val):.6f}")
    print(f"  lambda = {sol.lambda_(rho_val, t_val):.6f}")

    # Show metric tensor
    g = sol.metric_at(rho_val, t_val)
    print(f"\n  Metric tensor g_ij:")
    print(f"    {g}")

    # Show field tensor
    F = sol.field_tensor_at(rho_val, t_val)
    print(f"\n  Field tensor F_munu (non-zero components):")
    for i in range(4):
        for j in range(i+1, 4):
            if abs(F[i, j]) > 1e-10:
                print(f"    F[{i},{j}] = {F[i, j]:.6f}")

    # Verify field equations
    residuals = sol.verify_field_equations(rho_val, t_val)
    print(f"\n  Field equation residuals:")
    for name, value in residuals.items():
        print(f"    {name}: {value:.2e}")

    # Print symbolic expressions
    print(f"\n  Symbolic expressions:")
    print(f"    psi = {sol.psi_symbolic}")
    print(f"    mu  = {sol.mu_symbolic}")
    print(f"    lambda = {sol.lambda_symbolic}")


def demo_case_ii():
    """Demonstrate Case (ii): psi = 0 solutions."""
    print_section("Case (ii): psi = 0 (Only phi non-zero)")

    # Create solution with rho_only variant (equation 56)
    sol = CaseIISolution(
        variant='rho_only',
        m=1.0,
        n=0.5,
        a=1.0,
        p=0.0,
        q=0.0
    )
    print(f"\nSolution: {sol}")

    rho_val, t_val = 1.5, 0.5
    print(f"\nEvaluating at (rho={rho_val}, t={t_val}):")

    print(f"  phi = {sol.phi(rho_val, t_val):.6f}")
    print(f"  psi = {sol.psi(rho_val, t_val):.6f} (always 0 for Case ii)")
    print(f"  mu  = {sol.mu(rho_val, t_val):.6f}")
    print(f"  lambda = {sol.lambda_(rho_val, t_val):.6f}")

    # Metric at several rho values
    print(f"\n  g_tt component vs rho (at t={t_val}):")
    for r in [0.5, 1.0, 1.5, 2.0, 2.5]:
        g = sol.metric_at(r, t_val)
        print(f"    rho={r:.1f}: g_tt = {g[3,3]:.6f}")

    # Verify field equations
    residuals = sol.verify_field_equations(rho_val, t_val)
    print(f"\n  Field equation residuals:")
    for name, value in residuals.items():
        print(f"    {name}: {value:.2e}")


def demo_case_iii():
    """Demonstrate Case (iii): both phi and psi non-zero."""
    print_section("Case (iii): phi != 0, psi != 0 (Both non-zero)")

    # Create solution with variant_71 (equation 71)
    sol = CaseIIISolution(
        variant='variant_71',
        m=1.0,
        n=0.0,
        a=0.5,
        p=0.0,
        q=0.0,
        b=0.0
    )
    print(f"\nSolution: {sol}")

    rho_val, t_val = 2.0, 1.0
    print(f"\nEvaluating at (rho={rho_val}, t={t_val}):")

    print(f"  phi = {sol.phi(rho_val, t_val):.6f}")
    print(f"  psi = {sol.psi(rho_val, t_val):.6f}")
    print(f"  mu  = {sol.mu(rho_val, t_val):.6f}")
    print(f"  lambda = {sol.lambda_(rho_val, t_val):.6f}")

    # Full field tensor
    F = sol.field_tensor_at(rho_val, t_val)
    print(f"\n  Field tensor F_munu:")
    print(f"    {F}")

    # Energy tensor
    E = sol.energy_tensor_at(rho_val, t_val)
    print(f"\n  Energy tensor E^beta_alpha (diagonal components):")
    print(f"    E^1_1 = {E[0,0]:.6f}")
    print(f"    E^2_2 = {E[1,1]:.6f}")
    print(f"    E^3_3 = {E[2,2]:.6f}")
    print(f"    E^4_4 = {E[3,3]:.6f}")
    print(f"    Trace = {np.trace(E):.2e} (should be ~0 for EM field)")

    # Verify field equations
    residuals = sol.verify_field_equations(rho_val, t_val)
    print(f"\n  Field equation residuals:")
    for name, value in residuals.items():
        print(f"    {name}: {value:.2e}")


def demo_comparison():
    """Compare different solution variants."""
    print_section("Comparison of Solution Variants")

    # Create one solution from each case
    solutions = [
        ("Case (i) t_only", CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)),
        ("Case (ii) rho_only", CaseIISolution(variant='rho_only', m=1.0)),
        ("Case (iii) variant_71", CaseIIISolution(variant='variant_71', m=1.0, a=1.0)),
    ]

    rho_val, t_val = 1.5, 0.5

    print(f"\nValues at (rho={rho_val}, t={t_val}):")
    print(f"\n{'Solution':<25} {'phi':>10} {'psi':>10} {'mu':>10} {'lambda':>10}")
    print("-" * 70)

    for name, sol in solutions:
        phi = sol.phi(rho_val, t_val)
        psi = sol.psi(rho_val, t_val)
        mu = sol.mu(rho_val, t_val)
        lam = sol.lambda_(rho_val, t_val)
        print(f"{name:<25} {phi:>10.4f} {psi:>10.4f} {mu:>10.4f} {lam:>10.4f}")


def demo_bessel_solution():
    """Demonstrate the Bessel function solution for Case (i)."""
    print_section("Bessel Function Solution (Wave Equation)")

    # Create Bessel function solution
    sol = CaseISolution(
        alpha=1.0,
        beta=0.0,
        variant='bessel',
        k=1.0,      # Wave number
        A=0.5,      # Amplitude
        epsilon=0.0  # Phase
    )
    print(f"\nSolution: {sol}")
    print("  x = A * J_0(k*rho) * cos(k*t + epsilon)")

    # Evaluate at several points along rho axis
    t_val = 0.0
    print(f"\n  Values along rho axis at t={t_val}:")
    print(f"    {'rho':>6} {'psi':>10} {'mu':>10}")
    print("    " + "-" * 30)

    for rho_val in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        psi = sol.psi(rho_val, t_val)
        mu = sol.mu(rho_val, t_val)
        print(f"    {rho_val:>6.1f} {psi:>10.4f} {mu:>10.4f}")

    # Time evolution at fixed rho
    rho_val = 1.0
    print(f"\n  Time evolution at rho={rho_val}:")
    print(f"    {'t':>6} {'psi':>10} {'mu':>10}")
    print("    " + "-" * 30)

    for t_val in np.linspace(0, 2*np.pi, 7):
        psi = sol.psi(rho_val, t_val)
        mu = sol.mu(rho_val, t_val)
        print(f"    {t_val:>6.2f} {psi:>10.4f} {mu:>10.4f}")


def main():
    """Run all demonstrations."""
    print("\n" + "="*60)
    print("  EINSTEIN-MAXWELL CYLINDRICAL SYMMETRY SOLUTIONS")
    print("  Misra & Radhakrishna (1962)")
    print("="*60)

    demo_case_i()
    demo_case_ii()
    demo_case_iii()
    demo_comparison()
    demo_bessel_solution()

    print_section("Demo Complete")
    print("\nAll exact solutions demonstrated successfully!")
    print("These solutions represent electromagnetic fields in curved spacetime")
    print("satisfying the Einstein-Maxwell field equations in vacuo.")


if __name__ == '__main__':
    main()
