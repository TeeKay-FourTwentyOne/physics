"""
Demonstration of the PINN-based Quantum Hamilton-Jacobi solver.

This example trains a Physics-Informed Neural Network to learn the exact
quantum momentum function p(x) = -ix for the 1D harmonic oscillator ground state.

The quantum momentum satisfies the Riccati form of the QHJ equation:
    p^2 + i*hbar*(dp/dx) = 2m(E - V(x))
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from quantum_hj import HarmonicOscillator, PINNSolver


def main():
    print("=" * 60)
    print("PINN-based Quantum Hamilton-Jacobi Solver Demo")
    print("Learning the Quantum Momentum p(x) = -ix")
    print("=" * 60)
    print()

    # Create potential and solver
    potential = HarmonicOscillator(omega=1.0, mass=1.0)
    solver = PINNSolver(potential, hbar=1.0)

    print("System: V(x) = x^2/2  (natural units: hbar = m = omega = 1)")
    print("Exact quantum momentum for ground state: p(x) = -ix")
    print()

    # Training parameters
    n_epochs = 3000

    # Train for ground state
    print("Training PINN for ground state (E = 0.5)...")
    print("-" * 40)

    pinn = solver.train_for_energy(E=0.5, n_epochs=n_epochs, verbose=True)

    print()

    # Evaluate at test points
    test_points = [0, 1, 2, -1.5, 0.5 + 0.3j]

    print("Learned p(x) vs exact p(x) = -ix:")
    print("-" * 60)
    print(f"{'x':>15} | {'p_PINN':>20} | {'p_exact':>20} | {'error':>8}")
    print("-" * 60)

    for x in test_points:
        x_arr = np.array([complex(x)])
        p_pinn = pinn.evaluate(x_arr)[0]
        p_exact = -1j * x
        error = abs(p_pinn - p_exact)
        print(f"{str(x):>15} | {p_pinn.real:>8.4f}{p_pinn.imag:+8.4f}j | "
              f"{p_exact.real:>8.4f}{p_exact.imag:+8.4f}j | {error:>8.4f}")

    print()

    # Verify QHJ equation
    print("QHJ equation residual check (p^2 + i*dp/dx = 1 - x^2):")
    print("-" * 50)
    h = 1e-5
    for x in [0, 1, 2]:
        x_arr = np.array([float(x) + 0j])
        p = pinn.evaluate(x_arr)[0]
        dp_dx = (pinn.evaluate(np.array([x + h + 0j]))[0] -
                 pinn.evaluate(np.array([x - h + 0j]))[0]) / (2 * h)
        lhs = p**2 + 1j * dp_dx
        rhs = 1 - x**2
        print(f"x = {x}: |residual| = {abs(lhs - rhs):.6f}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
