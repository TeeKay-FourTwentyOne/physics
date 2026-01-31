"""
Tests for the N-dimensional DVR solver.

These tests validate the sparse DVR implementation for arbitrary dimensions.
"""

import pytest
import numpy as np

from quantum_hj.potentials_nd import (
    IsotropicOscillatorND, CoupledOscillator3D, CoupledOscillator4D
)
from quantum_hj.dvr_nd import DVRSolverND, compute_dvr_reference_nd


class TestDVRSolverND:
    """Test the N-D DVR solver class."""

    def test_creation_3d(self):
        """Test creating a 3D DVR solver."""
        pot = IsotropicOscillatorND(ndim=3)
        ranges = [(-5, 5), (-5, 5), (-5, 5)]

        dvr = DVRSolverND(pot, ranges, n_basis=10)

        assert dvr.ndim == 3
        assert dvr.total_basis == 10**3
        assert len(dvr.grids) == 3
        assert len(dvr.spacings) == 3

    def test_creation_4d(self):
        """Test creating a 4D DVR solver."""
        pot = IsotropicOscillatorND(ndim=4)
        ranges = [(-5, 5)] * 4

        dvr = DVRSolverND(pot, ranges, n_basis=8)

        assert dvr.ndim == 4
        assert dvr.total_basis == 8**4
        assert len(dvr.grids) == 4

    def test_scalar_n_basis(self):
        """Test that scalar n_basis is expanded to all dimensions."""
        pot = IsotropicOscillatorND(ndim=3)
        ranges = [(-5, 5)] * 3

        dvr = DVRSolverND(pot, ranges, n_basis=15)

        assert dvr.n_basis == [15, 15, 15]

    def test_list_n_basis(self):
        """Test that list n_basis works."""
        pot = IsotropicOscillatorND(ndim=3)
        ranges = [(-5, 5)] * 3

        dvr = DVRSolverND(pot, ranges, n_basis=[10, 12, 14])

        assert dvr.n_basis == [10, 12, 14]
        assert dvr.total_basis == 10 * 12 * 14

    def test_grid_creation(self):
        """Test that grids are created correctly."""
        pot = IsotropicOscillatorND(ndim=2)
        ranges = [(-4, 4), (-6, 6)]

        dvr = DVRSolverND(pot, ranges, n_basis=20)

        # Check grid bounds
        assert dvr.grids[0][0] > -4
        assert dvr.grids[0][-1] < 4
        assert dvr.grids[1][0] > -6
        assert dvr.grids[1][-1] < 6


class TestKineticMatrix:
    """Test the kinetic energy matrix construction."""

    def test_kinetic_matrix_symmetric(self):
        """Test that 1D kinetic matrix is symmetric."""
        pot = IsotropicOscillatorND(ndim=1)
        dvr = DVRSolverND(pot, [(-5, 5)], n_basis=20)

        T = dvr._kinetic_matrix_1d(20, dvr.spacings[0])

        # Check symmetry
        diff = T - T.T
        assert np.max(np.abs(diff.toarray())) < 1e-10

    def test_kinetic_matrix_diagonal_positive(self):
        """Test that diagonal elements are positive."""
        pot = IsotropicOscillatorND(ndim=1)
        dvr = DVRSolverND(pot, [(-5, 5)], n_basis=20)

        T = dvr._kinetic_matrix_1d(20, dvr.spacings[0])
        diag = T.diagonal()

        assert np.all(diag > 0)


class TestHamiltonianConstruction:
    """Test the Hamiltonian construction."""

    def test_hamiltonian_sparse(self):
        """Test that Hamiltonian is sparse."""
        pot = IsotropicOscillatorND(ndim=3)
        ranges = [(-5, 5)] * 3

        dvr = DVRSolverND(pot, ranges, n_basis=10)
        H = dvr.build_hamiltonian()

        # Should be sparse
        from scipy import sparse
        assert sparse.issparse(H)

        # Shape should match total_basis
        assert H.shape == (1000, 1000)

    def test_hamiltonian_symmetric(self):
        """Test that Hamiltonian is symmetric."""
        pot = IsotropicOscillatorND(ndim=2)
        ranges = [(-5, 5)] * 2

        dvr = DVRSolverND(pot, ranges, n_basis=15)
        H = dvr.build_hamiltonian()

        # Check symmetry
        diff = H - H.T
        assert np.max(np.abs(diff.toarray())) < 1e-10


class TestSolve:
    """Test the eigenvalue solver."""

    def test_solve_returns_correct_shape(self):
        """Test that solve returns correct shapes."""
        pot = IsotropicOscillatorND(ndim=2)
        ranges = [(-6, 6)] * 2

        dvr = DVRSolverND(pot, ranges, n_basis=20)
        energies, wavefunctions = dvr.solve(n_states=5)

        assert len(energies) == 5
        assert wavefunctions.shape == (400, 5)

    def test_energies_sorted(self):
        """Test that energies are sorted ascending."""
        pot = IsotropicOscillatorND(ndim=2)
        ranges = [(-6, 6)] * 2

        dvr = DVRSolverND(pot, ranges, n_basis=20)
        energies, _ = dvr.solve(n_states=10)

        # Check sorted
        assert np.all(energies[1:] >= energies[:-1])

    def test_2d_isotropic_reference(self):
        """Test 2D isotropic oscillator against known energies."""
        pot = IsotropicOscillatorND(ndim=2, omega=1.0)
        ranges = [(-6, 6)] * 2

        dvr = DVRSolverND(pot, ranges, n_basis=40)
        energies, _ = dvr.solve(n_states=6)

        # Expected: 1.0, 2.0, 2.0, 3.0, 3.0, 3.0
        expected = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0]

        for E_dvr, E_exact in zip(energies, expected):
            rel_error = abs(E_dvr - E_exact) / E_exact
            assert rel_error < 0.01, f"Energy error {rel_error:.1%}"


class TestIsotropicOscillator3D:
    """Test 3D isotropic oscillator eigenvalues."""

    def test_ground_state(self):
        """Test ground state energy E = 3/2."""
        pot = IsotropicOscillatorND(ndim=3, omega=1.0)
        ranges = [(-6, 6)] * 3

        dvr = DVRSolverND(pot, ranges, n_basis=25)
        energies, _ = dvr.solve(n_states=5)

        exact_E0 = 1.5
        rel_error = abs(energies[0] - exact_E0) / exact_E0

        assert rel_error < 0.01, f"Ground state error {rel_error:.1%}"

    def test_first_excited_degeneracy(self):
        """Test first excited states have degeneracy 3."""
        pot = IsotropicOscillatorND(ndim=3, omega=1.0)
        ranges = [(-6, 6)] * 3

        dvr = DVRSolverND(pot, ranges, n_basis=25)
        energies, _ = dvr.solve(n_states=5)

        # E1 = 5/2 with 3-fold degeneracy
        exact_E1 = 2.5

        # States 1, 2, 3 should all be ~2.5
        for i in [1, 2, 3]:
            rel_error = abs(energies[i] - exact_E1) / exact_E1
            assert rel_error < 0.01, f"State {i} error {rel_error:.1%}"


class TestCoupledOscillator3D:
    """Test 3D coupled oscillator."""

    def test_ground_state(self):
        """Test ground state energy matches normal mode formula."""
        pot = CoupledOscillator3D(lambda1=0.2, lambda2=0.15)
        ranges = [(-6, 6)] * 3

        dvr = DVRSolverND(pot, ranges, n_basis=25)
        energies, _ = dvr.solve(n_states=5)

        exact_E0 = pot.exact_energy((0, 0, 0))
        rel_error = abs(energies[0] - exact_E0) / exact_E0

        assert rel_error < 0.01, f"Ground state error {rel_error:.1%}"

    def test_coupling_shifts_energy(self):
        """Test that coupling changes energies from uncoupled case."""
        pot_coupled = CoupledOscillator3D(lambda1=0.2, lambda2=0.15)
        pot_uncoupled = IsotropicOscillatorND(ndim=3)
        ranges = [(-6, 6)] * 3

        dvr_coupled = DVRSolverND(pot_coupled, ranges, n_basis=25)
        dvr_uncoupled = DVRSolverND(pot_uncoupled, ranges, n_basis=25)

        E_coupled, _ = dvr_coupled.solve(n_states=5)
        E_uncoupled, _ = dvr_uncoupled.solve(n_states=5)

        # Energies should be different
        assert not np.allclose(E_coupled, E_uncoupled, rtol=0.001)


class TestIsotropicOscillator4D:
    """Test 4D isotropic oscillator eigenvalues."""

    def test_ground_state(self):
        """Test ground state energy E = 2.0."""
        pot = IsotropicOscillatorND(ndim=4, omega=1.0)
        ranges = [(-6, 6)] * 4

        # Use smaller basis for 4D
        dvr = DVRSolverND(pot, ranges, n_basis=15)
        energies, _ = dvr.solve(n_states=5)

        exact_E0 = 2.0
        rel_error = abs(energies[0] - exact_E0) / exact_E0

        assert rel_error < 0.02, f"4D ground state error {rel_error:.1%}"

    def test_first_excited_degeneracy(self):
        """Test first excited states have degeneracy 4."""
        pot = IsotropicOscillatorND(ndim=4, omega=1.0)
        ranges = [(-6, 6)] * 4

        dvr = DVRSolverND(pot, ranges, n_basis=15)
        energies, _ = dvr.solve(n_states=6)

        # E1 = 3.0 with 4-fold degeneracy
        exact_E1 = 3.0

        # States 1, 2, 3, 4 should all be ~3.0
        for i in [1, 2, 3, 4]:
            rel_error = abs(energies[i] - exact_E1) / exact_E1
            assert rel_error < 0.02, f"4D state {i} error {rel_error:.1%}"


class TestCoupledOscillator4D:
    """Test 4D coupled oscillator."""

    def test_ground_state(self):
        """Test ground state energy."""
        pot = CoupledOscillator4D(coupling=0.2)
        ranges = [(-6, 6)] * 4

        dvr = DVRSolverND(pot, ranges, n_basis=15)
        energies, _ = dvr.solve(n_states=5)

        exact_E0 = pot.exact_energy((0, 0, 0, 0))
        rel_error = abs(energies[0] - exact_E0) / exact_E0

        assert rel_error < 0.02, f"4D coupled ground state error {rel_error:.1%}"


class TestMemoryEstimate:
    """Test memory estimation."""

    def test_memory_estimate_3d(self):
        """Test memory estimate for 3D system."""
        pot = IsotropicOscillatorND(ndim=3)
        ranges = [(-5, 5)] * 3

        dvr = DVRSolverND(pot, ranges, n_basis=30)
        mem_mb = dvr.memory_estimate_mb()

        # Should be reasonable (not gigabytes for small basis)
        assert mem_mb > 0
        assert mem_mb < 1000  # Less than 1GB

    def test_memory_estimate_4d(self):
        """Test memory estimate for 4D system."""
        pot = IsotropicOscillatorND(ndim=4)
        ranges = [(-5, 5)] * 4

        dvr = DVRSolverND(pot, ranges, n_basis=15)
        mem_mb = dvr.memory_estimate_mb()

        assert mem_mb > 0
        assert mem_mb < 2000  # Less than 2GB


class TestWavefunctionReshape:
    """Test wavefunction reshaping."""

    def test_reshape_3d(self):
        """Test reshaping to 3D grid."""
        pot = IsotropicOscillatorND(ndim=3)
        ranges = [(-6, 6)] * 3

        dvr = DVRSolverND(pot, ranges, n_basis=10)
        dvr.solve(n_states=1)

        psi_3d = dvr.get_wavefunction_nd(0)

        assert psi_3d.shape == (10, 10, 10)

    def test_reshape_4d(self):
        """Test reshaping to 4D grid."""
        pot = IsotropicOscillatorND(ndim=4)
        ranges = [(-6, 6)] * 4

        dvr = DVRSolverND(pot, ranges, n_basis=8)
        dvr.solve(n_states=1)

        psi_4d = dvr.get_wavefunction_nd(0)

        assert psi_4d.shape == (8, 8, 8, 8)


class TestConvenienceFunction:
    """Test the convenience function."""

    def test_compute_dvr_reference_nd(self):
        """Test the convenience function."""
        pot = IsotropicOscillatorND(ndim=3)
        ranges = [(-6, 6)] * 3

        energies = compute_dvr_reference_nd(
            pot, ranges, n_states=5, n_basis=20
        )

        assert len(energies) == 5
        assert energies[0] < energies[1]  # Sorted


class TestIndexConversion:
    """Test multi-index to linear index conversion."""

    def test_linear_to_multi(self):
        """Test linear to multi-index conversion."""
        pot = IsotropicOscillatorND(ndim=3)
        dvr = DVRSolverND(pot, [(-5, 5)] * 3, n_basis=5)

        # Index 0 should be (0, 0, 0)
        assert dvr._linear_to_multi_index(0) == (0, 0, 0)

        # Index 1 should be (0, 0, 1)
        assert dvr._linear_to_multi_index(1) == (0, 0, 1)

        # Index 5 should be (0, 1, 0)
        assert dvr._linear_to_multi_index(5) == (0, 1, 0)

    def test_multi_to_linear(self):
        """Test multi to linear index conversion."""
        pot = IsotropicOscillatorND(ndim=3)
        dvr = DVRSolverND(pot, [(-5, 5)] * 3, n_basis=5)

        assert dvr._multi_to_linear_index((0, 0, 0)) == 0
        assert dvr._multi_to_linear_index((0, 0, 1)) == 1
        assert dvr._multi_to_linear_index((0, 1, 0)) == 5

    def test_roundtrip(self):
        """Test that conversion is consistent."""
        pot = IsotropicOscillatorND(ndim=3)
        dvr = DVRSolverND(pot, [(-5, 5)] * 3, n_basis=5)

        for idx in range(dvr.total_basis):
            multi = dvr._linear_to_multi_index(idx)
            back = dvr._multi_to_linear_index(multi)
            assert back == idx
