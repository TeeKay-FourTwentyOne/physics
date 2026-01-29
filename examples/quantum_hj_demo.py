"""
Demonstration of the Quantum Hamilton-Jacobi solver.

This example computes the energy eigenvalues for a 1D harmonic oscillator
using the Leacock-Padgett formalism and compares them to the exact values.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from quantum_hj import QuantumHJSolver, HarmonicOscillator


def main():
    print("=" * 60)
    print("Quantum Hamilton-Jacobi Solver Demo")
    print("Leacock-Padgett Formalism for 1D Harmonic Oscillator")
    print("=" * 60)
    print()

    # Create potential and solver
    potential = HarmonicOscillator(omega=1.0, mass=1.0)
    solver = QuantumHJSolver(potential, hbar=1.0)

    print("System: V(x) = x^2/2  (natural units: hbar = m = omega = 1)")
    print("Exact energies: E_n = n + 1/2")
    print()

    # Compute energies
    n_max = 9
    print(f"Computing first {n_max + 1} energy levels...")
    print()

    energies = solver.find_energies(n_max)
    exact = potential.exact_energies(n_max)

    # Display results
    print(f"{'n':>3}  {'Computed':>12}  {'Exact':>10}  {'Error (%)':>10}")
    print("-" * 42)

    for n in range(n_max + 1):
        E_comp = energies[n]
        E_exact = exact[n]
        error_pct = 100 * abs(E_comp - E_exact) / E_exact
        print(f"{n:>3}  {E_comp:>12.6f}  {E_exact:>10.1f}  {error_pct:>10.4f}")

    print()

    # Validation summary
    result = solver.validate(energies)
    max_error = result['max_relative_error'] * 100

    print(f"Maximum relative error: {max_error:.4f}%")
    print(f"Validation status: {result['status'].upper()}")
    print()

    # Show energy spacing
    print("Energy level spacing (should be ~1.0):")
    spacings = np.diff(energies)
    for n in range(len(spacings)):
        print(f"  E_{n+1} - E_{n} = {spacings[n]:.6f}")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
