"""
Demonstration of the 2D PINN quantum Hamilton-Jacobi solver.

This script validates the 2D PINN implementation against exact solutions
for the 2D harmonic oscillator, testing both isotropic and anisotropic cases.
"""

import numpy as np
import sys
import os

# Add parent directory to path for imports when running from examples/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_hj import HarmonicOscillator2D, PINNSolver2D


def demo_ground_state_momentum():
    """Demonstrate momentum learning for ground state."""
    print("=" * 60)
    print("2D PINN Quantum Hamilton-Jacobi Solver Demo")
    print("=" * 60)
    print()
    print("Part 1: Ground State Momentum (Isotropic omega_x = omega_y = 1)")
    print("-" * 60)

    potential = HarmonicOscillator2D(omega_x=1.0, omega_y=1.0)
    solver = PINNSolver2D(potential)

    # Ground state energy
    E = potential.exact_energy(0, 0)
    print(f"Ground state energy E = {E:.4f}")
    print()

    # Train PINN
    print("Training PINN (5000 epochs)...")
    pinn = solver.train_for_energy(E=E, n_epochs=5000, verbose=True)
    print()

    # Spot-check momentum values
    print("Spot-check momentum values (exact: p_x = -ix, p_y = -iy):")
    print("-" * 60)
    test_points = [
        (0.0, 0.0, 0.0, 0.0),         # p_x = 0, p_y = 0
        (1.0, 0.0, -1j, 0.0),         # p_x = -i, p_y = 0
        (0.0, 1.0, 0.0, -1j),         # p_x = 0, p_y = -i
        (1.0, 1.0, -1j, -1j),         # p_x = -i, p_y = -i
        (0.5, 0.0, -0.5j, 0.0),       # p_x = -0.5i, p_y = 0
    ]

    for x_val, y_val, p_x_exact, p_y_exact in test_points:
        x = np.array([x_val + 0j])
        y = np.array([y_val + 0j])
        p_x, p_y = pinn.evaluate(x, y)
        p_x, p_y = p_x[0], p_y[0]

        err_x = abs(p_x - p_x_exact)
        err_y = abs(p_y - p_y_exact)

        print(f"  ({x_val}, {y_val}): p_x = {p_x:.4f} (exact: {p_x_exact}), "
              f"p_y = {p_y:.4f} (exact: {p_y_exact})")
        print(f"          errors: |delta_p_x| = {err_x:.4f}, |delta_p_y| = {err_y:.4f}")

    print()


def demo_action_integrals_isotropic():
    """Demonstrate action integral computation for isotropic oscillator."""
    print("Part 2: Action Integrals (Isotropic omega_x = omega_y = 1)")
    print("-" * 60)

    potential = HarmonicOscillator2D(omega_x=1.0, omega_y=1.0)
    solver = PINNSolver2D(potential)

    # Test states for isotropic oscillator
    states = [
        (0, 0),  # E = 1.0, J_x/ℏ = 0.5, J_y/ℏ = 0.5
        (1, 0),  # E = 2.0, J_x/ℏ = 1.5, J_y/ℏ = 0.5
        (0, 1),  # E = 2.0, J_x/ℏ = 0.5, J_y/ℏ = 1.5
        (1, 1),  # E = 3.0, J_x/ℏ = 1.5, J_y/ℏ = 1.5
    ]

    print("\nValidation Table (Isotropic):")
    print("-" * 70)
    print(f"{'State':<12} {'E':>8} {'J_x/hbar':>10} {'exp':>6} {'J_y/hbar':>10} {'exp':>6} {'Status':<8}")
    print("-" * 70)

    for n_x, n_y in states:
        E = potential.exact_energy(n_x, n_y)
        result = solver.verify_state(E, n_x, n_y, n_epochs=5000, verbose=False, tol=0.1)

        status = "PASS" if result['passed'] else "FAIL"
        print(f"({n_x}, {n_y}){'':<6} {E:>8.4f} "
              f"{result['J_x_over_hbar']:>10.4f} {n_x + 0.5:>6.1f} "
              f"{result['J_y_over_hbar']:>10.4f} {n_y + 0.5:>6.1f} "
              f"{status:<8}")

    print()


def demo_action_integrals_anisotropic():
    """Demonstrate action integrals for anisotropic oscillator."""
    print("Part 3: Action Integrals (Anisotropic omega_x = 1, omega_y = sqrt(2))")
    print("-" * 60)

    omega_y = np.sqrt(2)
    potential = HarmonicOscillator2D(omega_x=1.0, omega_y=omega_y)
    solver = PINNSolver2D(potential)

    # Test states for anisotropic oscillator
    # E_{n_x, n_y} = (n_x + 0.5) + √2*(n_y + 0.5)
    states = [
        (0, 0),  # E ≈ 1.207
        (1, 0),  # E ≈ 2.207
        (0, 1),  # E ≈ 2.621
    ]

    print("\nValidation Table (Anisotropic):")
    print("-" * 70)
    print(f"{'State':<12} {'E':>8} {'J_x/hbar':>10} {'exp':>6} {'J_y/hbar':>10} {'exp':>6} {'Status':<8}")
    print("-" * 70)

    for n_x, n_y in states:
        E = potential.exact_energy(n_x, n_y)
        result = solver.verify_state(E, n_x, n_y, n_epochs=5000, verbose=False, tol=0.1)

        status = "PASS" if result['passed'] else "FAIL"
        print(f"({n_x}, {n_y}){'':<6} {E:>8.4f} "
              f"{result['J_x_over_hbar']:>10.4f} {n_x + 0.5:>6.1f} "
              f"{result['J_y_over_hbar']:>10.4f} {n_y + 0.5:>6.1f} "
              f"{status:<8}")

    print()


def demo_complex_plane_momentum():
    """Demonstrate momentum evaluation in complex plane."""
    print("Part 4: Complex Plane Momentum Values")
    print("-" * 60)

    potential = HarmonicOscillator2D()
    solver = PINNSolver2D(potential)
    E = potential.exact_energy(0, 0)

    print("Training PINN for ground state...")
    pinn = solver.train_for_energy(E=E, n_epochs=5000, verbose=False)

    # Test at complex points
    print("\nMomentum at complex positions (exact: p = -iz):")
    print("-" * 60)

    complex_points = [
        (0.5 + 0.2j, 0.0),       # p_x = 0.2 - 0.5j
        (0.0, 0.5 + 0.2j),       # p_y = 0.2 - 0.5j
        (1.0 + 0.5j, 1.0 - 0.3j),  # p_x = 0.5 - 1.0j, p_y = -0.3 - 1.0j
    ]

    for x_val, y_val in complex_points:
        x = np.array([x_val])
        y = np.array([y_val])
        p_x, p_y = pinn.evaluate(x, y)
        p_x, p_y = p_x[0], p_y[0]

        p_x_exact = -1j * x_val
        p_y_exact = -1j * y_val

        err_x = abs(p_x - p_x_exact)
        err_y = abs(p_y - p_y_exact)

        print(f"  x = {x_val}, y = {y_val}")
        print(f"    p_x = {p_x:.4f} (exact: {p_x_exact:.4f}, err: {err_x:.4f})")
        print(f"    p_y = {p_y:.4f} (exact: {p_y_exact:.4f}, err: {err_y:.4f})")

    print()


def main():
    """Run all demos."""
    try:
        demo_ground_state_momentum()
        demo_action_integrals_isotropic()
        demo_action_integrals_anisotropic()
        demo_complex_plane_momentum()

        print("=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure PyTorch is installed: pip install torch")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
