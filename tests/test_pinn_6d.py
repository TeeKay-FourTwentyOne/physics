"""
Tests for 6D PINN quantum Hamilton-Jacobi solver.

These tests validate the LP+PINN method in 6D where grid-based DVR
is impractical (would require >8TB memory).

Reference validation uses:
- Exact analytical solutions for separable oscillator
- Normal mode analysis for coupled harmonic systems
"""

import pytest
import numpy as np
import torch

from quantum_hj import (
    IsotropicOscillatorND,
    CoupledOscillator6D,
    TriatomicVibrational,
    normal_mode_ground_state_energy,
    normal_mode_energy,
)
from quantum_hj.pinn_nd import QuantumNumberTrainerND


class TestPotentials6D:
    """Test 6D potential classes."""

    def test_isotropic_oscillator_6d_energy(self):
        """Test exact energy for 6D isotropic oscillator."""
        pot = IsotropicOscillatorND(ndim=6)

        # Ground state: E = 6 * 0.5 = 3.0
        E_ground = pot.exact_energy((0, 0, 0, 0, 0, 0))
        assert abs(E_ground - 3.0) < 1e-10

        # First excited (one quantum): E = 3.0 + 1.0 = 4.0
        E_1 = pot.exact_energy((1, 0, 0, 0, 0, 0))
        assert abs(E_1 - 4.0) < 1e-10

    def test_isotropic_oscillator_6d_potential(self):
        """Test potential evaluation."""
        pot = IsotropicOscillatorND(ndim=6)

        # At origin
        V_origin = pot(0, 0, 0, 0, 0, 0)
        assert abs(V_origin) < 1e-10

        # At unit point
        V_unit = pot(1, 1, 1, 1, 1, 1)
        assert abs(V_unit - 3.0) < 1e-10  # 0.5 * 6 = 3.0

    def test_isotropic_oscillator_6d_alpha_matrix(self):
        """Test alpha matrix for exact momentum."""
        pot = IsotropicOscillatorND(ndim=6)
        alpha = pot.alpha_matrix()

        # Should be identity for omega=1, mass=1
        assert alpha.shape == (6, 6)
        np.testing.assert_allclose(alpha, np.eye(6), atol=1e-10)

    def test_coupled_oscillator_6d_creation(self):
        """Test 6D coupled oscillator creation."""
        pot = CoupledOscillator6D(coupling=0.15)

        assert pot.ndim == 6
        assert pot.coupling == 0.15

    def test_coupled_oscillator_6d_eigenvalues(self):
        """Test eigenvalues of circulant Hessian."""
        pot = CoupledOscillator6D(coupling=0.15)

        # For circulant matrix, eigenvalues are 1 + 2*lambda*cos(2*pi*k/6)
        # k=0: 1 + 2*0.15*cos(0) = 1.30
        # k=1: 1 + 2*0.15*cos(pi/3) = 1.15
        # k=2: 1 + 2*0.15*cos(2*pi/3) = 0.85
        # k=3: 1 + 2*0.15*cos(pi) = 0.70
        # k=4: 1 + 2*0.15*cos(4*pi/3) = 0.85
        # k=5: 1 + 2*0.15*cos(5*pi/3) = 1.15
        expected = np.array([1.30, 1.15, 0.85, 0.70, 0.85, 1.15])
        actual = pot._eigenvalues

        np.testing.assert_allclose(np.sort(actual), np.sort(expected), atol=1e-10)

    def test_coupled_oscillator_6d_ground_energy(self):
        """Test ground state energy from normal modes."""
        pot = CoupledOscillator6D(coupling=0.15)

        # Ground state: E = (1/2) * sum(omega_i)
        E_ground = pot.exact_energy((0, 0, 0, 0, 0, 0))

        # Compare with direct calculation
        omega = np.sqrt(pot._eigenvalues)
        E_expected = 0.5 * np.sum(omega)

        assert abs(E_ground - E_expected) < 1e-10

    def test_coupled_oscillator_6d_potential(self):
        """Test potential evaluation with periodic coupling."""
        pot = CoupledOscillator6D(coupling=0.15)

        # At origin
        V_origin = pot(0, 0, 0, 0, 0, 0)
        assert abs(V_origin) < 1e-10

        # At unit point: V = 0.5*6 + 0.15*6 = 3.9
        V_unit = pot(1, 1, 1, 1, 1, 1)
        assert abs(V_unit - 3.9) < 1e-10

    def test_coupled_oscillator_6d_hessian(self):
        """Test Hessian is circulant."""
        pot = CoupledOscillator6D(coupling=0.15)
        H = pot.hessian_at_minimum()

        # Check structure
        assert H.shape == (6, 6)

        # Diagonal should be 1
        np.testing.assert_allclose(np.diag(H), np.ones(6), atol=1e-10)

        # Off-diagonal nearest neighbors should be 0.15
        for i in range(6):
            j = (i + 1) % 6
            assert abs(H[i, j] - 0.15) < 1e-10
            assert abs(H[j, i] - 0.15) < 1e-10

    def test_triatomic_creation(self):
        """Test triatomic vibrational model creation."""
        pot = TriatomicVibrational()

        assert pot.ndim == 6
        assert pot.k_stretch == 1.0
        assert pot.k_bend_in == 0.5
        assert pot.k_bend_out == 0.3

    def test_triatomic_ground_energy(self):
        """Test triatomic ground state energy."""
        pot = TriatomicVibrational()
        E_ground = pot.exact_energy((0, 0, 0, 0, 0, 0))

        # Should be positive and reasonable
        assert E_ground > 0
        assert E_ground < 5  # Rough bound

    def test_invalid_coupling(self):
        """Test that invalid coupling raises error."""
        with pytest.raises(ValueError):
            CoupledOscillator6D(coupling=0.6)  # |lambda| >= 0.5 is invalid


class TestNormalModeReference:
    """Test normal mode reference solvers."""

    def test_ground_state_energy_isotropic(self):
        """Test normal mode energy for isotropic oscillator."""
        H = np.eye(6)  # Unit Hessian
        E = normal_mode_ground_state_energy(H)

        # E = 0.5 * sum(omega) = 0.5 * 6 = 3.0
        assert abs(E - 3.0) < 1e-10

    def test_ground_state_energy_coupled(self):
        """Test normal mode energy for coupled oscillator."""
        pot = CoupledOscillator6D(coupling=0.15)
        H = pot.hessian_at_minimum()

        E_from_hessian = normal_mode_ground_state_energy(H)
        E_from_potential = pot.exact_energy((0, 0, 0, 0, 0, 0))

        assert abs(E_from_hessian - E_from_potential) < 1e-10

    def test_excited_state_energy(self):
        """Test normal mode energy for excited states."""
        H = np.eye(6)

        # First excited state
        E_1 = normal_mode_energy(H, (1, 0, 0, 0, 0, 0))
        assert abs(E_1 - 4.0) < 1e-10  # 3.0 + 1.0

        # Doubly excited
        E_2 = normal_mode_energy(H, (1, 1, 0, 0, 0, 0))
        assert abs(E_2 - 5.0) < 1e-10  # 3.0 + 2.0


class TestTrainer6D:
    """Test 6D trainer configuration."""

    def test_dimension_aware_defaults(self):
        """Test that 6D gets appropriate defaults."""
        pot = IsotropicOscillatorND(ndim=6)
        trainer = QuantumNumberTrainerND(pot, (0, 0, 0, 0, 0, 0))

        # Check network structure
        n_layers = len([m for m in trainer.net.smooth_net.modules()
                        if isinstance(m, torch.nn.Linear)])
        assert n_layers == 8 + 1  # 8 hidden + 1 output

        # Check parameter count is in expected range
        n_params = sum(p.numel() for p in trainer.net.parameters())
        assert n_params > 1_000_000  # Should be > 1M
        assert n_params < 10_000_000  # Should be < 10M

    def test_importance_sampling(self):
        """Test importance sampling for 6D."""
        pot = IsotropicOscillatorND(ndim=6)
        trainer = QuantumNumberTrainerND(pot, (0, 0, 0, 0, 0, 0))

        # Sample with importance sampling
        coords = trainer.sample_collocation_points(1000, importance_sampling=True)

        assert len(coords) == 6
        for c in coords:
            assert len(c) > 900  # Should have most of the requested points

        # Check that some points are near origin (from Gaussian sampling)
        origin_dist = torch.sqrt(sum(torch.abs(c)**2 for c in coords))
        near_origin = (origin_dist < 2.0).sum().item()
        assert near_origin > 20  # Should have points near origin

    def test_supervision_loss_6d(self):
        """Test supervision loss runs for 6D."""
        pot = IsotropicOscillatorND(ndim=6)
        trainer = QuantumNumberTrainerND(pot, (0, 0, 0, 0, 0, 0))

        # Should compute without error
        loss = trainer.supervision_loss(n_points=25)
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestPINN6DSeparable:
    """Test 6D PINN on separable oscillator (quick validation)."""

    @pytest.mark.slow
    def test_separable_ground_state_quick(self):
        """Quick test of 6D separable ground state (reduced epochs)."""
        pot = IsotropicOscillatorND(ndim=6)
        trainer = QuantumNumberTrainerND(pot, (0, 0, 0, 0, 0, 0))

        # Quick training (reduced from full 30000 epochs)
        result = trainer.train(
            n_epochs=2000,
            n_collocation=2000,
            supervision_weight=20.0,
            verbose=False
        )

        E_pinn = result['E']
        E_exact = 3.0

        # With reduced training, allow larger error
        rel_error = abs(E_pinn - E_exact) / E_exact
        assert rel_error < 0.10, f"Energy error {rel_error*100:.1f}% > 10%"

    @pytest.mark.slow
    def test_separable_momentum_structure(self):
        """Test momentum has correct structure after training."""
        pot = IsotropicOscillatorND(ndim=6)
        trainer = QuantumNumberTrainerND(pot, (0, 0, 0, 0, 0, 0))

        # Quick training
        trainer.train(n_epochs=1000, n_collocation=1000, verbose=False)

        # Evaluate at (1, 0, 0, 0, 0, 0)
        coords = [np.array([1.0 + 0j])] + [np.array([0.0 + 0j])] * 5
        p = trainer.evaluate(*coords)

        # p_0 should be approximately -i at x_0 = 1
        p0 = p[0][0]
        assert abs(p0.imag + 1.0) < 0.3, f"p_0 = {p0}, expected ~-i"

        # Other components should be small
        for i in range(1, 6):
            assert abs(p[i][0]) < 0.3, f"p_{i} = {p[i][0]}, expected ~0"


class TestPINN6DCoupled:
    """Test 6D PINN on coupled oscillator."""

    @pytest.mark.slow
    def test_coupled_ground_state_quick(self):
        """Quick test of 6D coupled ground state."""
        pot = CoupledOscillator6D(coupling=0.15)
        trainer = QuantumNumberTrainerND(pot, (0, 0, 0, 0, 0, 0))

        # Reference energy from normal modes
        E_exact = pot.exact_energy((0, 0, 0, 0, 0, 0))

        # Quick training
        result = trainer.train(
            n_epochs=2000,
            n_collocation=2000,
            supervision_weight=20.0,
            verbose=False
        )

        E_pinn = result['E']
        rel_error = abs(E_pinn - E_exact) / E_exact

        # With reduced training, allow larger error
        assert rel_error < 0.15, f"Energy error {rel_error*100:.1f}% > 15%"


# Markers for slow tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
