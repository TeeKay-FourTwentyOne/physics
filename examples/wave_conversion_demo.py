#!/usr/bin/env python3
"""
Wave Conversion Demo

Demonstrates the gravitational-electromagnetic wave mode conversion
module based on Mishima & Tomizawa (Phys. Rev. D 110, 024038, 2024).

This module uses the harmonic mapping method to study nonlinear
mode conversion between gravitational and electromagnetic waves
in cylindrically symmetric spacetimes.

Key features demonstrated:
1. Creating solutions with different seeds and submanifolds
2. Computing Ernst potentials and metric functions
3. C-energy decomposition into gravitational and EM modes
4. Mode occupancy ratios (R_grav, R_em)
5. Conversion to Misra-Radhakrishna variables
"""

import numpy as np

from em_cylindrical.wave_conversion import (
    WaveConversion,
    LinearWaveSeed,
    SolitonicSeed,
    ComplexLine,
    LagrangianPlane,
    CEnergy,
)
from em_cylindrical.wave_conversion.conversion import create_conversion_demo


def demo_linear_wave_complex_line():
    """Demonstrate linear wave seed with complex line submanifold."""
    print("=" * 60)
    print("Demo 1: Linear Wave Seed + Complex Line")
    print("=" * 60)

    # Create linear wave seed
    # tau = A * J_0(k*rho) * cos(k*t + epsilon)
    seed = LinearWaveSeed(
        amplitude=0.5,
        wavenumber=1.0,
        phase=0.0,
        A_param=0.1,  # Controls imaginary part
    )
    print(f"\nSeed: LinearWaveSeed(amplitude=0.5, wavenumber=1.0)")

    # Create complex line submanifold
    # xi = cos(2*theta) * z, eta = sin(2*theta) * z
    # theta controls GW/EM mixing
    theta = np.pi / 6  # 30 degrees
    submanifold = ComplexLine(theta=theta)
    print(f"Submanifold: ComplexLine(theta={theta:.4f} rad = {np.degrees(theta):.1f} deg)")

    # Create wave conversion solution
    wc = WaveConversion(seed=seed, submanifold=submanifold)

    # Test point
    rho, t = 2.0, 1.0
    print(f"\nEvaluating at (rho, t) = ({rho}, {t}):")

    # Get Ernst potentials
    E, F = wc.ernst_potentials(rho, t)
    print(f"\n  Ernst Potentials:")
    print(f"    E = {E.real:.6f} + {E.imag:.6f}i")
    print(f"    F = {F.real:.6f} + {F.imag:.6f}i")

    # Get reduced potentials
    xi = wc.xi(rho, t)
    eta = wc.eta(rho, t)
    print(f"\n  Reduced Potentials:")
    print(f"    xi  = {xi.real:.6f} + {xi.imag:.6f}i")
    print(f"    eta = {eta.real:.6f} + {eta.imag:.6f}i")
    print(f"    |xi|^2 + |eta|^2 = {wc.constraint_value(rho, t):.6f} (< 1 required)")

    # Get metric functions
    funcs = wc.metric_functions(rho, t)
    print(f"\n  Metric Functions:")
    print(f"    psi   = {funcs['psi']:.6f}")
    print(f"    gamma = {funcs['gamma']:.6f}")
    print(f"    A_z   = {funcs['A_z']:.6f}")
    print(f"    chi   = {funcs['chi']:.6f}")

    # C-energy analysis
    c = wc.c_energy()
    R_grav, R_em = c.occupancy_ratios(rho, t)
    print(f"\n  Mode Occupancy:")
    print(f"    Gravitational: {R_grav:.1%}")
    print(f"    Electromagnetic: {R_em:.1%}")

    # Theoretical prediction for complex line
    print(f"\n  Theoretical (for complex line):")
    print(f"    cos^2(2*theta) = {np.cos(2*theta)**2:.3f} (grav)")
    print(f"    sin^2(2*theta) = {np.sin(2*theta)**2:.3f} (EM)")


def demo_solitonic_lagrangian():
    """Demonstrate solitonic seed with Lagrangian plane submanifold."""
    print("\n" + "=" * 60)
    print("Demo 2: Solitonic Seed + Lagrangian Plane")
    print("=" * 60)
    print("\nThis configuration produces NONTRIVIAL mode conversions")
    print("that persist to null infinity.")

    # Create solitonic seed with constraint q^2 - p^2 - l^2 = 1
    # Using p=1, l=1 -> q = sqrt(3)
    seed = SolitonicSeed.from_p_l(p=1.0, l=1.0)
    print(f"\nSeed: SolitonicSeed(p={seed.p}, q={seed.q:.4f}, l={seed.l})")
    print(f"  Constraint check: q^2 - p^2 - l^2 = {seed.q**2 - seed.p**2 - seed.l**2:.6f} (should be 1)")

    # Create Lagrangian plane submanifold
    submanifold = LagrangianPlane()
    print(f"Submanifold: LagrangianPlane()")

    # Create wave conversion solution
    wc = WaveConversion(seed=seed, submanifold=submanifold)

    # Conversion ratio
    ratio = wc.conversion_ratio()
    bounds = seed.conversion_amplification_bounds()
    print(f"\n  EM Amplification Ratio: {ratio:.3f}")
    print(f"  Theoretical bounds: [{bounds[0]:.1f}, {bounds[1]:.1f}]")

    # Test at several points
    print(f"\n  Evaluating at multiple points:")
    test_points = [(1.5, 1.5), (2.0, 2.0), (2.5, 2.5), (3.0, 3.0)]

    for rho, t in test_points:
        try:
            c = wc.c_energy()
            R_grav, R_em = c.occupancy_ratios(rho, t)
            print(f"    (rho={rho}, t={t}): R_grav={R_grav:.1%}, R_em={R_em:.1%}")
        except Exception as e:
            print(f"    (rho={rho}, t={t}): [coordinate transform issue]")


def demo_conversion_to_misra_radhakrishna():
    """Demonstrate conversion to Misra-Radhakrishna variables."""
    print("\n" + "=" * 60)
    print("Demo 3: Conversion to Misra-Radhakrishna Variables")
    print("=" * 60)

    # Create a simple wave conversion solution
    wc = create_conversion_demo('linear_complex')
    print("\nUsing: create_conversion_demo('linear_complex')")

    # Test point
    rho, t = 2.0, 1.0

    # Get MR variables
    mr = wc.to_misra_radhakrishna(rho, t)

    print(f"\nAt (rho, t) = ({rho}, {t}):")
    print(f"\n  Kompaneets-Jordan-Ehlers form:")
    funcs = wc.metric_functions(rho, t)
    print(f"    psi   = {funcs['psi']:.6f}")
    print(f"    gamma = {funcs['gamma']:.6f}")

    print(f"\n  Misra-Radhakrishna form:")
    print(f"    mu     = {mr['mu']:.6f}  (= 2*psi)")
    print(f"    lambda = {mr['lambda']:.6f}  (= 2*gamma)")
    print(f"    phi    = {mr['phi']:.6f}  (EM azimuthal)")
    print(f"    psi_MR = {mr['psi_MR']:.6f}  (EM axial)")


def demo_c_energy_modes():
    """Demonstrate C-energy mode decomposition."""
    print("\n" + "=" * 60)
    print("Demo 4: C-Energy Mode Decomposition")
    print("=" * 60)

    # Create solution with interesting dynamics
    seed = LinearWaveSeed(amplitude=0.3, wavenumber=0.5)
    submanifold = ComplexLine(theta=np.pi / 8)
    wc = WaveConversion(seed=seed, submanifold=submanifold)

    print("\nC-energy provides a local, conserved measure of energy")
    print("for cylindrically symmetric spacetimes (Thorne 1965).")

    c = wc.c_energy()

    # Evaluate at test point
    rho, t = 2.0, 0.5

    print(f"\nAt (rho, t) = ({rho}, {t}):")

    # Total energy within radius
    E_total = c.total_energy(rho, t)
    print(f"\n  Total C-energy within rho={rho}: {E_total:.6f}")

    # Local density and flux
    density = c.density(rho, t)
    flux = c.flux(rho, t)
    print(f"  Energy density: {density:.6f}")
    print(f"  Energy flux: {flux:.6f}")

    # Mode decomposition
    modes = c.mode_decomposition(rho, t)
    print(f"\n  Mode decomposition:")
    print(f"    Gravitational +: {modes['grav_plus']:.6e}")
    print(f"    Gravitational x: {modes['grav_cross']:.6e}")
    print(f"    EM z-mode:       {modes['em_z']:.6e}")
    print(f"    EM phi-mode:     {modes['em_phi']:.6e}")
    print(f"    Total:           {modes['total']:.6e}")

    # Occupancy ratios
    R_grav, R_em = c.occupancy_ratios(rho, t)
    print(f"\n  Mode occupancy:")
    print(f"    R_grav = {R_grav:.3f} ({R_grav*100:.1f}%)")
    print(f"    R_em   = {R_em:.3f} ({R_em*100:.1f}%)")
    print(f"    Sum    = {R_grav + R_em:.6f} (should be 1)")

    # Conservation check
    residual = c.verify_conservation(rho, t)
    print(f"\n  Conservation law residual: {residual:.2e}")
    print(f"    (should be ~0 for valid solutions)")


def demo_all_configurations():
    """Show all available demo configurations."""
    print("\n" + "=" * 60)
    print("Demo 5: Available Configurations")
    print("=" * 60)

    configs = [
        'solitonic_lagrangian',  # Nontrivial conversion
        'solitonic_complex',
        'linear_complex',
        'linear_lagrangian',
    ]

    print("\nAvailable configurations via create_conversion_demo():")
    for config in configs:
        wc = create_conversion_demo(config)
        print(f"\n  '{config}':")
        print(f"    Seed: {type(wc.seed).__name__}")
        print(f"    Submanifold: {type(wc.submanifold).__name__}")

        # Show key property
        if isinstance(wc.seed, SolitonicSeed):
            print(f"    Parameters: p={wc.seed.p}, q={wc.seed.q:.3f}, l={wc.seed.l}")
        elif isinstance(wc.seed, LinearWaveSeed):
            print(f"    Parameters: A={wc.seed.amplitude}, k={wc.seed.wavenumber}")

        if isinstance(wc.submanifold, ComplexLine):
            print(f"    Mixing angle: {np.degrees(wc.submanifold.theta):.1f} deg")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 60)
    print("# Gravitational-Electromagnetic Wave Mode Conversion Demo")
    print("#" * 60)
    print("\nBased on: Mishima & Tomizawa")
    print("Phys. Rev. D 110, 024038 (2024)")
    print("arXiv:2405.04231")

    demo_linear_wave_complex_line()
    demo_solitonic_lagrangian()
    demo_conversion_to_misra_radhakrishna()
    demo_c_energy_modes()
    demo_all_configurations()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
