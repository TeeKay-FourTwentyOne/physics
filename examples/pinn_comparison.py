"""
PINN Quantum Momentum Accuracy Comparison.

This script demonstrates that the PINN learns the exact quantum momentum
p(x) = -ix for the harmonic oscillator ground state.

Note: The PINN learns the analytic solution p = -ix which satisfies the
QHJ Riccati equation. The Leacock-Padgett contour integral quantization
uses the classical momentum with its branch structure, which is different.
"""

import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from quantum_hj import HarmonicOscillator, PINNSolver


def main():
    print("=" * 70)
    print("PINN Quantum Momentum Accuracy")
    print("Learning p(x) = -ix for Harmonic Oscillator")
    print("=" * 70)
    print()

    # Create potential and solver
    potential = HarmonicOscillator(omega=1.0, mass=1.0)
    solver = PINNSolver(potential, hbar=1.0)

    # Training parameters
    n_epochs = 3000

    print("Training PINN for ground state (E = 0.5)...")
    t0 = time.time()
    pinn = solver.train_for_energy(E=0.5, n_epochs=n_epochs, verbose=False)
    train_time = time.time() - t0
    print(f"Training completed in {train_time:.1f} seconds")
    print()

    # Test points
    test_points = [0, 1, 2, -1.5, 0.5 + 0.3j, 1 + 1j, -2 + 0.5j]

    print("=" * 70)
    print("Learned p(x) vs Exact p(x) = -ix")
    print("=" * 70)
    print()
    print(f"{'x':>15} | {'p_PINN':>22} | {'p_exact':>22} | {'error':>10}")
    print("-" * 75)

    errors = []
    for x in test_points:
        x_arr = np.array([complex(x)])
        p_pinn = pinn.evaluate(x_arr)[0]
        p_exact = -1j * x
        error = abs(p_pinn - p_exact)
        errors.append(error)
        print(f"{str(x):>15} | {p_pinn.real:>9.4f}{p_pinn.imag:+9.4f}j | "
              f"{p_exact.real:>9.4f}{p_exact.imag:+9.4f}j | {error:>10.6f}")

    print()
    print("=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    print()
    print(f"Mean error:     {np.mean(errors):.6f}")
    print(f"Max error:      {np.max(errors):.6f}")
    print(f"Physics loss:   {pinn.loss_history[-1]:.2e}")
    print(f"Training time:  {train_time:.1f} s")
    print()

    # QHJ residual check
    print("QHJ Equation Residual (should be ~0):")
    print("-" * 50)
    h = 1e-5
    for x in [0, 1, 2, 0.5 + 0.3j]:
        x_arr = np.array([complex(x)])
        p = pinn.evaluate(x_arr)[0]
        dp_dx = (pinn.evaluate(np.array([x + h]))[0] -
                 pinn.evaluate(np.array([x - h]))[0]) / (2 * h)
        V = 0.5 * x * x
        lhs = p**2 + 1j * dp_dx
        rhs = 2 * (0.5 - V)
        print(f"x = {str(x):>12}: |residual| = {abs(lhs - rhs):.6f}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
