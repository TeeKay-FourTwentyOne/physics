"""
Tests for the DVR (Discrete Variable Representation) reference solver.

Validates DVR accuracy against:
1. 2D harmonic oscillator (analytical solution)
2. Coupled harmonic oscillator (analytical via normal modes)
"""

import numpy as np
import pytest

from quantum_hj.dvr import DVRSolver2D, compute_dvr_reference
from quantum_hj.potentials_2d import (
    HarmonicOscillator2D,
    CoupledHarmonicOscillator,
    HenonHeiles
)


class TestDVRSolver2D:
    """Tests for DVRSolver2D class."""

    def test_creation(self):
        """Test DVR solver instantiation."""
        pot = HarmonicOscillator2D()
        dvr = DVRSolver2D(pot, x_range=(-6, 6), y_range=(-6, 6), n_basis_x=40, n_basis_y=40)

        assert dvr.n_basis_x == 40
        assert dvr.n_basis_y == 40
        assert len(dvr.x_grid) == 40
        assert len(dvr.y_grid) == 40

    def test_grid_spacing(self):
        """Test grid spacing calculation."""
        pot = HarmonicOscillator2D()
        dvr = DVRSolver2D(pot, x_range=(-5, 5), y_range=(-5, 5), n_basis_x=50, n_basis_y=50)

        expected_dx = 10.0 / 51  # (x_max - x_min) / (n + 1)
        assert abs(dvr.dx - expected_dx) < 1e-10
        assert abs(dvr.dy - expected_dx) < 1e-10

    def test_hamiltonian_symmetry(self):
        """Test that Hamiltonian is symmetric."""
        pot = HarmonicOscillator2D()
        dvr = DVRSolver2D(pot, x_range=(-4, 4), y_range=(-4, 4), n_basis_x=20)
        H = dvr.build_hamiltonian()

        # Convert to dense for symmetry check
        H_dense = H.toarray()
        assert np.allclose(H_dense, H_dense.T)


class TestDVRHarmonicOscillator:
    """Test DVR against 2D harmonic oscillator analytical results."""

    def test_isotropic_ground_state(self):
        """Test ground state energy for isotropic oscillator."""
        pot = HarmonicOscillator2D(omega_x=1.0, omega_y=1.0)
        dvr = DVRSolver2D(pot, x_range=(-8, 8), y_range=(-8, 8), n_basis_x=60)

        energies, _ = dvr.solve(n_states=5)

        # Ground state: E_0 = ℏ(ω_x/2 + ω_y/2) = 1.0
        assert abs(energies[0] - 1.0) < 0.01

    def test_isotropic_first_excited(self):
        """Test first excited states for isotropic oscillator."""
        pot = HarmonicOscillator2D(omega_x=1.0, omega_y=1.0)
        dvr = DVRSolver2D(pot, x_range=(-8, 8), y_range=(-8, 8), n_basis_x=60)

        energies, _ = dvr.solve(n_states=5)

        # First excited states: E = 2.0 (doubly degenerate)
        # States (1,0) and (0,1) both have E = 1.5 + 0.5 = 2.0
        assert abs(energies[1] - 2.0) < 0.01
        assert abs(energies[2] - 2.0) < 0.01

    def test_isotropic_multiple_states(self):
        """Test multiple states for isotropic oscillator."""
        pot = HarmonicOscillator2D(omega_x=1.0, omega_y=1.0)
        dvr = DVRSolver2D(pot, x_range=(-10, 10), y_range=(-10, 10), n_basis_x=80)

        energies, _ = dvr.solve(n_states=10)

        # Expected energies: 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, ...
        expected = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0]
        for i in range(min(6, len(energies))):
            assert abs(energies[i] - expected[i]) < 0.02, \
                f"State {i}: DVR={energies[i]:.4f}, expected={expected[i]}"

    def test_anisotropic_ground_state(self):
        """Test ground state for anisotropic oscillator."""
        pot = HarmonicOscillator2D(omega_x=1.0, omega_y=2.0)
        dvr = DVRSolver2D(pot, x_range=(-8, 8), y_range=(-6, 6), n_basis_x=60)

        energies, _ = dvr.solve(n_states=5)

        # Ground state: E_0 = 0.5*1 + 0.5*2 = 1.5
        assert abs(energies[0] - 1.5) < 0.01

    def test_anisotropic_spectrum(self):
        """Test spectrum for anisotropic oscillator."""
        pot = HarmonicOscillator2D(omega_x=1.0, omega_y=1.5)
        dvr = DVRSolver2D(pot, x_range=(-8, 8), y_range=(-7, 7), n_basis_x=60)

        energies, _ = dvr.solve(n_states=6)

        # Expected energies:
        # (0,0): 0.5 + 0.75 = 1.25
        # (1,0): 1.5 + 0.75 = 2.25
        # (0,1): 0.5 + 2.25 = 2.75
        # (2,0): 2.5 + 0.75 = 3.25
        # (1,1): 1.5 + 2.25 = 3.75
        expected = [1.25, 2.25, 2.75, 3.25, 3.75]

        for i in range(min(5, len(energies))):
            assert abs(energies[i] - expected[i]) < 0.02, \
                f"State {i}: DVR={energies[i]:.4f}, expected={expected[i]}"


class TestDVRCoupledOscillator:
    """Test DVR against coupled harmonic oscillator analytical results."""

    def test_weak_coupling_ground_state(self):
        """Test ground state with weak coupling."""
        coupling = 0.2
        pot = CoupledHarmonicOscillator(coupling=coupling)
        dvr = DVRSolver2D(pot, x_range=(-8, 8), y_range=(-8, 8), n_basis_x=60)

        energies, _ = dvr.solve(n_states=5)

        # Exact ground state: E_0 = 0.5*(ω+ + ω-) where ω± = √(1 ± λ)
        omega_plus = np.sqrt(1 + coupling)
        omega_minus = np.sqrt(1 - coupling)
        E_exact = 0.5 * (omega_plus + omega_minus)

        assert abs(energies[0] - E_exact) < 0.01, \
            f"Ground state: DVR={energies[0]:.4f}, exact={E_exact:.4f}"

    def test_moderate_coupling_spectrum(self):
        """Test spectrum with moderate coupling."""
        coupling = 0.5
        pot = CoupledHarmonicOscillator(coupling=coupling)
        dvr = DVRSolver2D(pot, x_range=(-10, 10), y_range=(-10, 10), n_basis_x=80)

        energies, _ = dvr.solve(n_states=10)

        # Compute exact spectrum
        omega_plus = np.sqrt(1 + coupling)
        omega_minus = np.sqrt(1 - coupling)

        exact_energies = []
        for n_plus in range(4):
            for n_minus in range(4):
                E = omega_plus * (n_plus + 0.5) + omega_minus * (n_minus + 0.5)
                exact_energies.append(E)
        exact_energies.sort()

        for i in range(min(6, len(energies))):
            assert abs(energies[i] - exact_energies[i]) < 0.02, \
                f"State {i}: DVR={energies[i]:.4f}, exact={exact_energies[i]:.4f}"


class TestDVRHenonHeiles:
    """Test DVR for Henon-Heiles potential (no analytical solution)."""

    def test_ground_state_bounded(self):
        """Test that ground state exists below escape energy."""
        coupling = 0.1
        pot = HenonHeiles(coupling=coupling)
        dvr = DVRSolver2D(pot, x_range=(-8, 8), y_range=(-8, 8), n_basis_x=60)

        energies, _ = dvr.solve(n_states=5)

        # Ground state should be close to harmonic value (1.0) for weak coupling
        assert 0.9 < energies[0] < 1.1
        assert energies[0] < pot.escape_energy

    def test_bound_states_below_escape(self):
        """Test that all computed bound states are below escape energy."""
        coupling = 0.1
        pot = HenonHeiles(coupling=coupling)
        dvr = DVRSolver2D(pot, x_range=(-10, 10), y_range=(-10, 10), n_basis_x=80)

        energies, _ = dvr.solve(n_states=20)

        # All bound states should be below escape energy
        escape_E = pot.escape_energy  # 1/(6*0.1^2) ≈ 16.67
        bound_energies = [E for E in energies if E < escape_E]

        assert len(bound_energies) >= 10, \
            f"Expected at least 10 bound states, found {len(bound_energies)}"

    def test_henon_heiles_near_harmonic_at_low_energy(self):
        """Test that low-lying states are close to harmonic values."""
        coupling = 0.05  # Very weak coupling
        pot = HenonHeiles(coupling=coupling)
        dvr = DVRSolver2D(pot, x_range=(-10, 10), y_range=(-10, 10), n_basis_x=80)

        energies, _ = dvr.solve(n_states=10)

        # For weak coupling, should be close to harmonic: 1, 2, 2, 3, 3, 3, ...
        harmonic_E = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0]

        for i in range(len(harmonic_E)):
            # Allow larger tolerance for higher states due to anharmonicity
            tol = 0.05 * (i + 1)
            assert abs(energies[i] - harmonic_E[i]) < tol, \
                f"State {i}: DVR={energies[i]:.4f}, harmonic={harmonic_E[i]}"


class TestComputeDVRReference:
    """Test the convenience function."""

    def test_convenience_function(self):
        """Test compute_dvr_reference helper."""
        pot = HarmonicOscillator2D()
        energies = compute_dvr_reference(
            pot, x_range=(-6, 6), y_range=(-6, 6),
            n_states=5, n_basis=50
        )

        assert len(energies) == 5
        assert abs(energies[0] - 1.0) < 0.02


class TestDVRWavefunction:
    """Test wavefunction extraction and probability density."""

    def test_wavefunction_shape(self):
        """Test wavefunction reshaping."""
        pot = HarmonicOscillator2D()
        dvr = DVRSolver2D(pot, x_range=(-6, 6), y_range=(-6, 6),
                          n_basis_x=30, n_basis_y=30)
        dvr.solve(n_states=3)

        psi_2d = dvr.get_wavefunction_2d(0)
        assert psi_2d.shape == (30, 30)

    def test_ground_state_symmetric(self):
        """Test ground state wavefunction symmetry for isotropic potential."""
        pot = HarmonicOscillator2D(omega_x=1.0, omega_y=1.0)
        dvr = DVRSolver2D(pot, x_range=(-6, 6), y_range=(-6, 6),
                          n_basis_x=40, n_basis_y=40)
        dvr.solve(n_states=3)

        psi_2d = dvr.get_wavefunction_2d(0)

        # Ground state should be approximately symmetric
        # |ψ(x,y)|² ≈ |ψ(y,x)|²
        prob = np.abs(psi_2d)**2
        prob_T = np.abs(psi_2d.T)**2

        # Normalize for comparison
        prob = prob / np.sum(prob)
        prob_T = prob_T / np.sum(prob_T)

        assert np.allclose(prob, prob_T, rtol=0.1)

    def test_probability_density_normalized(self):
        """Test that probability density integrates to ~1."""
        pot = HarmonicOscillator2D()
        dvr = DVRSolver2D(pot, x_range=(-8, 8), y_range=(-8, 8),
                          n_basis_x=50, n_basis_y=50)
        dvr.solve(n_states=3)

        # DVR eigenvectors are already normalized: sum |ψ|² = 1
        # Not: integral |ψ|² dx dy = 1 (need to multiply by dx*dy for that)
        psi_2d = dvr.get_wavefunction_2d(0)
        norm_sq = np.sum(np.abs(psi_2d)**2)

        # DVR eigenvectors are orthonormal under discrete sum
        assert 0.9 < norm_sq < 1.1


class TestDVRExpectationValue:
    """Test expectation value calculations."""

    def test_x_squared_expectation(self):
        """Test <x²> for harmonic oscillator ground state."""
        pot = HarmonicOscillator2D(omega_x=1.0, omega_y=1.0)
        dvr = DVRSolver2D(pot, x_range=(-10, 10), y_range=(-10, 10),
                          n_basis_x=80, n_basis_y=80)
        dvr.solve(n_states=3)

        # For DVR, expectation value is computed with discrete normalization
        # <x²> = sum_ij |ψ(x_i, y_j)|² x_i² (already normalized under discrete sum)
        psi_2d = dvr.get_wavefunction_2d(0)

        x2_sum = 0.0
        for i, x in enumerate(dvr.x_grid):
            for j, y in enumerate(dvr.y_grid):
                x2_sum += np.abs(psi_2d[i, j])**2 * x**2

        # <x²> = ℏ/(2mω) = 0.5 for ground state with m=ω=ℏ=1
        # Allow reasonable tolerance
        assert abs(x2_sum - 0.5) < 0.15, f"<x²> = {x2_sum:.4f}, expected ~0.5"
