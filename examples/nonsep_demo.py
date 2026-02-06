"""
Non-separable 2D Quantum Hamilton-Jacobi Solver Demonstration.

This script demonstrates the Leacock-Padgett PINN approach for non-separable
2D potentials where V(x,y) ≠ V_x(x) + V_y(y). It shows:

1. DVR (Discrete Variable Representation) reference calculations
2. Energy-conditioned PINN training
3. EBK quantization to find bound state energies
4. Comparison between PINN and DVR results

Test systems:
- Coupled Harmonic Oscillator: V = (1/2)(x² + y²) + λxy
- Henon-Heiles Potential: V = (1/2)(x² + y²) + λ(x²y - y³/3)

Usage:
    python examples/nonsep_demo.py

Reference:
    Leacock, R. A. & Padgett, M. J. (1983). Phys. Rev. Lett. 50, 3-6.
"""

import numpy as np
import time


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def demo_dvr_reference():
    """
    Demonstrate DVR reference solver for non-separable potentials.
    """
    print_header("DVR Reference Solver")

    from quantum_hj.dvr import DVRSolver2D, compute_dvr_reference
    from quantum_hj.potentials_2d import (
        HarmonicOscillator2D,
        CoupledHarmonicOscillator,
        HenonHeiles
    )

    # Test 1: Isotropic Harmonic Oscillator (validation)
    print("\n1. Isotropic Harmonic Oscillator (validation)")
    print("-" * 45)

    pot_ho = HarmonicOscillator2D(omega_x=1.0, omega_y=1.0)
    dvr_ho = DVRSolver2D(pot_ho, x_range=(-8, 8), y_range=(-8, 8), n_basis_x=60)

    start = time.time()
    energies_ho, _ = dvr_ho.solve(n_states=10)
    t_ho = time.time() - start

    print(f"  Computation time: {t_ho:.2f}s")
    print(f"  Grid size: {dvr_ho.n_basis_x}×{dvr_ho.n_basis_y} = {dvr_ho.n_basis_x**2} basis functions")
    print("\n  Energy levels (DVR vs Exact):")
    exact_ho = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0]
    for i in range(min(6, len(energies_ho))):
        error = abs(energies_ho[i] - exact_ho[i])
        print(f"    E_{i} = {energies_ho[i]:.5f} (exact: {exact_ho[i]:.1f}, error: {error:.1e})")

    # Test 2: Coupled Harmonic Oscillator
    print("\n2. Coupled Harmonic Oscillator: V = ½(x² + y²) + λxy")
    print("-" * 50)

    for coupling in [0.2, 0.5]:
        print(f"\n  Coupling λ = {coupling}")
        pot_coupled = CoupledHarmonicOscillator(coupling=coupling)

        # Compute exact normal mode energies
        omega_p = np.sqrt(1 + coupling)
        omega_m = np.sqrt(1 - coupling)
        print(f"    Normal mode frequencies: ω+ = {omega_p:.4f}, ω- = {omega_m:.4f}")

        exact_coupled = []
        for np_val in range(4):
            for nm_val in range(4):
                E = omega_p * (np_val + 0.5) + omega_m * (nm_val + 0.5)
                exact_coupled.append(E)
        exact_coupled.sort()

        # DVR calculation
        dvr_coupled = DVRSolver2D(pot_coupled, x_range=(-10, 10), y_range=(-10, 10),
                                   n_basis_x=60)
        energies_coupled, _ = dvr_coupled.solve(n_states=10)

        print("\n    DVR vs Exact:")
        for i in range(min(6, len(energies_coupled))):
            error = abs(energies_coupled[i] - exact_coupled[i])
            print(f"      E_{i} = {energies_coupled[i]:.5f} (exact: {exact_coupled[i]:.5f}, error: {error:.1e})")

    # Test 3: Henon-Heiles Potential
    print("\n3. Henon-Heiles Potential: V = ½(x² + y²) + λ(x²y - y³/3)")
    print("-" * 55)

    coupling_hh = 0.1
    pot_hh = HenonHeiles(coupling=coupling_hh)
    print(f"\n  Coupling λ = {coupling_hh}")
    print(f"  Escape energy E_escape = 1/(6λ²) = {pot_hh.escape_energy:.2f}")

    dvr_hh = DVRSolver2D(pot_hh, x_range=(-10, 10), y_range=(-10, 10), n_basis_x=70)
    energies_hh, _ = dvr_hh.solve(n_states=20)

    # Filter bound states
    bound_hh = [E for E in energies_hh if E < pot_hh.escape_energy]
    print(f"\n  Found {len(bound_hh)} bound states below E_escape")

    print("\n  Low-lying energy levels:")
    # Compare to harmonic (perturbative limit)
    harmonic_E = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0]
    for i in range(min(6, len(bound_hh))):
        deviation = bound_hh[i] - harmonic_E[i]
        print(f"    E_{i} = {bound_hh[i]:.5f} (harmonic: {harmonic_E[i]:.1f}, shift: {deviation:+.5f})")

    return {
        'coupled_0.2': compute_dvr_reference(
            CoupledHarmonicOscillator(0.2),
            x_range=(-10, 10), y_range=(-10, 10),
            n_states=10, n_basis=60
        ),
        'coupled_0.5': compute_dvr_reference(
            CoupledHarmonicOscillator(0.5),
            x_range=(-10, 10), y_range=(-10, 10),
            n_states=10, n_basis=60
        ),
        'henon_heiles': compute_dvr_reference(
            HenonHeiles(0.1),
            x_range=(-10, 10), y_range=(-10, 10),
            n_states=10, n_basis=70
        )
    }


def demo_pinn_training():
    """
    Demonstrate energy-conditioned PINN training.
    """
    print_header("Energy-Conditioned PINN Training")

    from quantum_hj.pinn_2d_nonsep import QuantumHJPINN2DNonSep
    from quantum_hj.potentials_2d import CoupledHarmonicOscillator

    # Train on coupled oscillator
    coupling = 0.2
    pot = CoupledHarmonicOscillator(coupling=coupling)

    print(f"\nPotential: V(x,y) = ½(x² + y²) + {coupling}xy")
    print(f"Energy range: [0.5, 5.0]")

    # Create and train PINN
    print("\nTraining PINN...")
    print("  Architecture: 5 layers × 256 neurons")
    print("  Input: (Re(x), Im(x), Re(y), Im(y), Re(E), Im(E))")
    print("  Output: (Re(p_x), Im(p_x), Re(p_y), Im(p_y))")

    pinn = QuantumHJPINN2DNonSep(
        pot,
        E_range=(0.5, 5.0),
        hidden_layers=5,
        hidden_size=256
    )

    start = time.time()
    final_loss = pinn.train(
        n_epochs=10000,
        lr=1e-3,
        n_collocation=1000,
        verbose=True
    )
    t_train = time.time() - start

    print(f"\nTraining completed in {t_train:.1f}s")
    print(f"Final loss: {final_loss:.2e}")

    # Verify physics residual at different energies
    print("\nPhysics residual at selected energies:")
    for E in [0.9, 1.5, 2.5, 3.5]:
        residual = pinn.verify_physics_residual(E, n_points=100)
        print(f"  E = {E:.1f}: residual = {residual:.2e}")

    return pinn


def demo_ebk_quantization(pinn, dvr_reference):
    """
    Demonstrate EBK quantization to find bound state energies.
    """
    print_header("EBK Quantization")

    from quantum_hj.pinn_2d_nonsep import EBKQuantizer
    from quantum_hj.potentials_2d import CoupledHarmonicOscillator

    pot = CoupledHarmonicOscillator(coupling=0.2)
    quant = EBKQuantizer(pinn, pot)

    print("\nEBK Quantization Conditions:")
    print("  J_x(E) = ℏ(n_x + ½)")
    print("  J_y(E) = ℏ(n_y + ½)")

    print("\nSearching for quantized energies...")

    # Find quantized energies for low-lying states
    results = []
    for n_x in range(3):
        for n_y in range(3):
            # Estimate search range
            E_est = pot.exact_energy(n_x, n_y)
            E_min = max(0.5, E_est * 0.8)
            E_max = min(5.0, E_est * 1.2)

            result = quant.find_quantized_energy(n_x, n_y, E_min, E_max, tol=0.1)
            results.append(result)

    # Sort by energy
    results.sort(key=lambda r: r['E'])

    print("\nFound quantized states:")
    print(f"{'State':^10} {'PINN E':^10} {'DVR E':^10} {'J_x/ℏ':^8} {'J_y/ℏ':^8} {'Error':^10}")
    print("-" * 60)

    dvr_sorted = sorted(dvr_reference)
    for i, result in enumerate(results[:6]):
        n_x, n_y = result['n_x'], result['n_y']
        E_pinn = result['E']
        E_dvr = dvr_sorted[i] if i < len(dvr_sorted) else np.nan
        error = abs(E_pinn - E_dvr) if i < len(dvr_sorted) else np.nan

        print(f"  ({n_x},{n_y})  {E_pinn:^10.4f} {E_dvr:^10.4f} "
              f"{result['J_x_over_hbar']:^8.3f} {result['J_y_over_hbar']:^8.3f} "
              f"{error:^10.4f}")

    return results


def demo_full_comparison():
    """
    Full comparison of PINN vs DVR for multiple potentials.
    """
    print_header("Full PINN vs DVR Comparison")

    from quantum_hj.pinn_2d_nonsep import NonSepPINNSolver
    from quantum_hj.potentials_2d import CoupledHarmonicOscillator, HenonHeiles
    from quantum_hj.dvr import compute_dvr_reference

    print("\nThis section compares energy-conditioned PINN results with")
    print("DVR reference calculations for non-separable potentials.")

    # Test 1: Coupled Oscillator (λ = 0.2)
    print("\n" + "-" * 50)
    print("Test 1: Coupled Oscillator (λ = 0.2)")
    print("-" * 50)

    pot1 = CoupledHarmonicOscillator(coupling=0.2)

    # Get DVR reference
    dvr1 = compute_dvr_reference(
        pot1, x_range=(-10, 10), y_range=(-10, 10),
        n_states=10, n_basis=60
    )

    # Get exact energies for comparison
    exact1 = []
    for n_p in range(4):
        for n_m in range(4):
            exact1.append(pot1.exact_energy(n_p, n_m))
    exact1 = sorted(exact1)[:10]

    print("\nDVR vs Exact (first 6 states):")
    for i in range(6):
        print(f"  E_{i}: DVR={dvr1[i]:.5f}, Exact={exact1[i]:.5f}, "
              f"error={abs(dvr1[i]-exact1[i]):.1e}")

    # Train PINN solver
    print("\nTraining PINN solver (this may take a few minutes)...")
    solver1 = NonSepPINNSolver(pot1, E_range=(0.5, 5.0))
    solver1.train(n_epochs=15000, verbose=False)

    # Compare
    comparison1 = solver1.compare_with_dvr(dvr1, n_states=6)
    print("\nPINN vs DVR comparison:")
    print(f"  Mean absolute error: {comparison1['mean_error']:.4f}")
    print(f"  Mean relative error: {comparison1['mean_rel_error']*100:.2f}%")
    print(f"  Max relative error:  {comparison1['max_rel_error']*100:.2f}%")

    # Test 2: Henon-Heiles (λ = 0.1)
    print("\n" + "-" * 50)
    print("Test 2: Henon-Heiles (λ = 0.1)")
    print("-" * 50)

    pot2 = HenonHeiles(coupling=0.1)
    print(f"Escape energy: {pot2.escape_energy:.2f}")

    # Get DVR reference
    dvr2 = compute_dvr_reference(
        pot2, x_range=(-10, 10), y_range=(-10, 10),
        n_states=10, n_basis=70
    )

    print("\nDVR energies (first 6 bound states):")
    for i in range(min(6, len(dvr2))):
        if dvr2[i] < pot2.escape_energy:
            print(f"  E_{i}: {dvr2[i]:.5f}")

    # Train PINN solver
    print("\nTraining PINN solver (this may take a few minutes)...")
    solver2 = NonSepPINNSolver(pot2, E_range=(0.5, 8.0))
    solver2.train(n_epochs=15000, verbose=False)

    # Compare
    comparison2 = solver2.compare_with_dvr(dvr2, n_states=6)
    print("\nPINN vs DVR comparison:")
    print(f"  Mean absolute error: {comparison2['mean_error']:.4f}")
    print(f"  Mean relative error: {comparison2['mean_rel_error']*100:.2f}%")
    print(f"  Max relative error:  {comparison2['max_rel_error']*100:.2f}%")

    return {'coupled': comparison1, 'henon_heiles': comparison2}


def main():
    """Run the full demonstration."""
    print("\n" + "=" * 60)
    print(" Non-Separable 2D Quantum Hamilton-Jacobi Solver")
    print(" Leacock-Padgett PINN Approach")
    print("=" * 60)

    # Part 1: DVR Reference
    dvr_refs = demo_dvr_reference()

    # Part 2: PINN Training (quick version for demo)
    print_header("Quick PINN Training Demo")
    print("\nNote: For full accuracy, use more training epochs.")
    print("This demo uses reduced parameters for faster execution.")

    from quantum_hj.pinn_2d_nonsep import QuantumHJPINN2DNonSep
    from quantum_hj.potentials_2d import CoupledHarmonicOscillator

    pot = CoupledHarmonicOscillator(coupling=0.2)
    pinn = QuantumHJPINN2DNonSep(
        pot,
        E_range=(0.5, 4.0),
        hidden_layers=4,
        hidden_size=128
    )

    print("\nTraining PINN (reduced epochs for demo)...")
    pinn.train(n_epochs=5000, verbose=True, n_collocation=500)

    # Part 3: EBK Quantization
    demo_ebk_quantization(pinn, dvr_refs['coupled_0.2'])

    # Summary
    print_header("Summary")
    print("""
The Leacock-Padgett PINN approach for non-separable 2D potentials:

1. DVR provides accurate reference eigenvalues using sparse matrix methods.
   - Sinc-DVR basis functions
   - Scales as O(N²) basis functions for N points per dimension

2. Energy-conditioned PINN learns p(x, y, E) across an energy range.
   - Single network covers multiple quantum states
   - Physics loss: QHJ equation residual
   - Curl loss: Irrotationality constraint

3. EBK quantization finds bound state energies.
   - Action integrals computed via contour integration
   - Simultaneous satisfaction of J_x = ℏ(n_x + ½), J_y = ℏ(n_y + ½)

Key advantages:
- Works for non-separable potentials where analytical solutions don't exist
- Provides both energies and momentum field (quantum current)
- Can handle potentials with complex phase-space structure

Limitations:
- Training time scales with energy range and accuracy requirements
- Contour integration may be sensitive to potential topology
""")


if __name__ == "__main__":
    main()
