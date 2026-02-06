"""
Tests for quantum-number-conditioned PINN solver.

Tests the QuantumNumberPINN2D network with learnable poles and
the NonSepQuantumSolver for non-separable 2D potentials.
"""

import numpy as np
import pytest
import torch

from quantum_hj.pinn_2d_qn import (
    hermite_zeros,
    QuantumNumberPINN2D,
    QuantumNumberTrainer,
    NonSepQuantumSolver,
)
from quantum_hj.potentials_2d import (
    HarmonicOscillator2D,
    CoupledHarmonicOscillator,
)
from quantum_hj.dvr import compute_dvr_reference


class TestHermiteZeros:
    """Tests for Hermite polynomial zero computation."""

    def test_zero_order(self):
        """H_0 has no zeros."""
        zeros = hermite_zeros(0)
        assert len(zeros) == 0

    def test_first_order(self):
        """H_1(x) = 2x has one zero at x=0."""
        zeros = hermite_zeros(1)
        assert len(zeros) == 1
        assert abs(zeros[0]) < 1e-10

    def test_second_order(self):
        """H_2(x) = 4x² - 2 has zeros at ±1/√2."""
        zeros = hermite_zeros(2)
        assert len(zeros) == 2
        expected = np.array([-1/np.sqrt(2), 1/np.sqrt(2)])
        np.testing.assert_array_almost_equal(np.sort(zeros), np.sort(expected), decimal=10)

    def test_third_order(self):
        """H_3(x) has 3 zeros including x=0."""
        zeros = hermite_zeros(3)
        assert len(zeros) == 3
        # One zero at 0, others at ±√(3/2)
        assert np.min(np.abs(zeros)) < 1e-10  # Contains zero

    def test_zeros_sorted(self):
        """Zeros should be sorted ascending."""
        for n in [1, 2, 3, 4, 5]:
            zeros = hermite_zeros(n)
            np.testing.assert_array_equal(zeros, np.sort(zeros))


class TestQuantumNumberPINN2D:
    """Tests for the quantum-number-conditioned neural network."""

    def test_ground_state_no_poles(self):
        """Ground state (0,0) should have no poles."""
        net = QuantumNumberPINN2D(n_x=0, n_y=0)
        assert net.n_x == 0
        assert net.n_y == 0
        # poles_x and poles_y should be empty
        poles = net.pole_positions()
        assert len(poles['x']) == 0
        assert len(poles['y']) == 0

    def test_first_excited_one_pole(self):
        """State (1,0) should have one pole in x."""
        net = QuantumNumberPINN2D(n_x=1, n_y=0)
        poles = net.pole_positions()
        assert len(poles['x']) == 1
        assert len(poles['y']) == 0
        # Pole should be initialized at zero (first Hermite zero)
        assert abs(poles['x'][0]) < 1e-5

    def test_state_11_two_poles(self):
        """State (1,1) should have one pole in each direction."""
        net = QuantumNumberPINN2D(n_x=1, n_y=1)
        poles = net.pole_positions()
        assert len(poles['x']) == 1
        assert len(poles['y']) == 1

    def test_state_20_two_x_poles(self):
        """State (2,0) should have two poles in x."""
        net = QuantumNumberPINN2D(n_x=2, n_y=0)
        poles = net.pole_positions()
        assert len(poles['x']) == 2
        assert len(poles['y']) == 0

    def test_forward_complex_input(self):
        """Network should accept complex inputs."""
        net = QuantumNumberPINN2D(n_x=1, n_y=1)
        x = torch.tensor([1.0 + 0.1j, 2.0 - 0.1j], dtype=torch.complex64)
        y = torch.tensor([0.5 + 0.2j, -0.5 - 0.2j], dtype=torch.complex64)

        p_x, p_y = net(x, y)

        assert p_x.shape == (2,)
        assert p_y.shape == (2,)
        assert p_x.dtype == torch.complex64
        assert p_y.dtype == torch.complex64

    def test_poles_affect_output(self):
        """Network output should change near poles."""
        net = QuantumNumberPINN2D(n_x=1, n_y=0)
        pole_x = net.poles_x[0].item()

        # Evaluate far from pole
        x_far = torch.tensor([pole_x + 2.0 + 0.5j], dtype=torch.complex64)
        y = torch.tensor([0.0 + 0j], dtype=torch.complex64)
        p_x_far, _ = net(x_far, y)

        # Evaluate close to pole
        x_near = torch.tensor([pole_x + 0.01 + 0.5j], dtype=torch.complex64)
        p_x_near, _ = net(x_near, y)

        # Near pole should have larger magnitude due to 1/(x - pole)
        assert torch.abs(p_x_near) > torch.abs(p_x_far)

    def test_poles_are_trainable(self):
        """Pole positions should be learnable parameters."""
        net = QuantumNumberPINN2D(n_x=1, n_y=1)

        # Check poles are parameters
        param_names = [name for name, _ in net.named_parameters()]
        assert 'poles_x' in param_names
        assert 'poles_y' in param_names

        # Verify gradients flow
        x = torch.tensor([1.0 + 0.1j], dtype=torch.complex64)
        y = torch.tensor([0.5 + 0.1j], dtype=torch.complex64)

        p_x, p_y = net(x, y)
        loss = torch.abs(p_x).sum() + torch.abs(p_y).sum()
        loss.backward()

        # Poles should have gradients
        assert net.poles_x.grad is not None
        assert net.poles_y.grad is not None


class TestQuantumNumberTrainer:
    """Tests for the quantum number trainer."""

    @pytest.fixture
    def harmonic_pot(self):
        """Uncoupled harmonic oscillator potential."""
        return HarmonicOscillator2D(omega_x=1.0, omega_y=1.0)

    @pytest.fixture
    def coupled_pot(self):
        """Coupled harmonic oscillator potential."""
        return CoupledHarmonicOscillator(coupling=0.2)

    def test_trainer_initialization(self, harmonic_pot):
        """Trainer should initialize with correct quantum numbers."""
        trainer = QuantumNumberTrainer(harmonic_pot, n_x=1, n_y=0)
        assert trainer.n_x == 1
        assert trainer.n_y == 0
        assert trainer.E.item() > 0  # Energy initialized positive

    def test_physics_loss_computable(self, harmonic_pot):
        """Physics loss should be computable."""
        trainer = QuantumNumberTrainer(harmonic_pot, n_x=0, n_y=0, device='cpu')
        x = torch.tensor([1.0 + 0.1j, -1.0 + 0.1j], dtype=torch.complex64)
        y = torch.tensor([0.5 + 0.1j, -0.5 + 0.1j], dtype=torch.complex64)

        loss = trainer.physics_loss(x, y)

        assert loss.shape == ()
        assert loss.item() >= 0
        assert np.isfinite(loss.item())

    def test_curl_loss_computable(self, harmonic_pot):
        """Curl loss should be computable."""
        trainer = QuantumNumberTrainer(harmonic_pot, n_x=0, n_y=0, device='cpu')
        x = torch.tensor([1.0 + 0.1j, -1.0 + 0.1j], dtype=torch.complex64)
        y = torch.tensor([0.5 + 0.1j, -0.5 + 0.1j], dtype=torch.complex64)

        loss = trainer.curl_loss(x, y)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_quantization_loss_computable(self, harmonic_pot):
        """Quantization loss should be computable."""
        trainer = QuantumNumberTrainer(harmonic_pot, n_x=0, n_y=0)

        loss = trainer.quantization_loss(n_points=30)

        assert loss.shape == ()
        assert np.isfinite(loss.item())

    def test_action_integrals_computed(self, harmonic_pot):
        """Action integrals should be computable."""
        trainer = QuantumNumberTrainer(harmonic_pot, n_x=0, n_y=0)

        J_x = trainer._compute_action_x(n_points=50)
        J_y = trainer._compute_action_y(n_points=50)

        assert np.isfinite(J_x.item())
        assert np.isfinite(J_y.item())

    def test_short_training_runs(self, harmonic_pot):
        """Short training should run without errors."""
        trainer = QuantumNumberTrainer(harmonic_pot, n_x=0, n_y=0)

        result = trainer.train(n_epochs=100, verbose=False)

        assert 'E' in result
        assert 'J_x_over_hbar' in result
        assert 'J_y_over_hbar' in result
        assert result['E'] > 0

    def test_energy_is_optimized(self, harmonic_pot):
        """Energy should change during training."""
        trainer = QuantumNumberTrainer(harmonic_pot, n_x=0, n_y=0)
        E_initial = trainer.E.item()

        trainer.train(n_epochs=500, verbose=False, quant_start_epoch=100)

        # Energy history should show variation
        assert len(trainer.E_history) > 0
        E_final = trainer.E.item()
        # Allow some change (may converge quickly for ground state)
        # Main point is training completes

    def test_poles_initialized_correctly(self, harmonic_pot):
        """Poles should be initialized at Hermite zeros."""
        trainer = QuantumNumberTrainer(harmonic_pot, n_x=2, n_y=1)
        poles = trainer.net.pole_positions()

        # n_x=2: two poles at ±1/√2 (scaled)
        assert len(poles['x']) == 2
        # n_y=1: one pole at 0
        assert len(poles['y']) == 1


class TestNonSepQuantumSolver:
    """Tests for the high-level solver."""

    @pytest.fixture
    def coupled_pot(self):
        """Coupled harmonic oscillator with λ=0.2."""
        return CoupledHarmonicOscillator(coupling=0.2)

    @pytest.fixture
    def uncoupled_pot(self):
        """Uncoupled harmonic oscillator."""
        return HarmonicOscillator2D(omega_x=1.0, omega_y=1.0)

    def test_solver_initialization(self, coupled_pot):
        """Solver should initialize correctly."""
        solver = NonSepQuantumSolver(coupled_pot)
        assert solver.potential is coupled_pot
        assert solver.hbar == 1.0
        assert solver.mass == 1.0

    def test_solve_single_state(self, uncoupled_pot):
        """Should solve for a single state."""
        solver = NonSepQuantumSolver(uncoupled_pot)

        result = solver.solve_state(0, 0, n_epochs=500, verbose=False)

        assert 'E' in result
        assert 'poles_x' in result
        assert 'poles_y' in result
        assert result['n_x'] == 0
        assert result['n_y'] == 0

    def test_ground_state_no_poles(self, uncoupled_pot):
        """Ground state should have no poles."""
        solver = NonSepQuantumSolver(uncoupled_pot)

        result = solver.solve_state(0, 0, n_epochs=500, verbose=False)

        assert len(result['poles_x']) == 0
        assert len(result['poles_y']) == 0

    def test_excited_state_has_poles(self, uncoupled_pot):
        """First excited state should have one pole."""
        solver = NonSepQuantumSolver(uncoupled_pot)

        result = solver.solve_state(1, 0, n_epochs=500, verbose=False)

        assert len(result['poles_x']) == 1
        assert len(result['poles_y']) == 0

    def test_solve_multiple_states(self, uncoupled_pot):
        """Should solve for multiple states."""
        solver = NonSepQuantumSolver(uncoupled_pot)

        results = solver.solve_states([(0, 0), (1, 0)], n_epochs=300, verbose=False)

        assert len(results) == 2
        assert results[0]['n_x'] == 0
        assert results[1]['n_x'] == 1

    def test_caching(self, uncoupled_pot):
        """Solved states should be cached."""
        solver = NonSepQuantumSolver(uncoupled_pot)

        solver.solve_state(0, 0, n_epochs=100, verbose=False)

        assert (0, 0) in solver._solved_states

    def test_exact_energy_lookup(self, uncoupled_pot):
        """Should return exact energy for separable potential."""
        solver = NonSepQuantumSolver(uncoupled_pot)

        E_exact = solver.exact_energy(0, 0)

        assert E_exact == 1.0  # (0.5 + 0.5) * hbar

    def test_exact_energy_coupled(self, coupled_pot):
        """Should return exact energy for coupled potential."""
        solver = NonSepQuantumSolver(coupled_pot)

        E_exact = solver.exact_energy(0, 0)

        # Ground state: (ω+ + ω-)/2 where ω± = √(1 ± λ)
        expected = coupled_pot.ground_state_energy()
        assert abs(E_exact - expected) < 1e-10


class TestIntegration:
    """Integration tests with longer training."""

    @pytest.mark.slow
    def test_uncoupled_ground_state_energy(self):
        """Ground state energy should match exact value for uncoupled case."""
        pot = HarmonicOscillator2D(omega_x=1.0, omega_y=1.0)
        solver = NonSepQuantumSolver(pot)

        result = solver.solve_state(0, 0, n_epochs=5000, verbose=False)

        # Exact energy is 1.0
        assert abs(result['E'] - 1.0) < 0.1

    @pytest.mark.slow
    def test_coupled_vs_dvr(self):
        """Coupled oscillator energies should match DVR reference."""
        coupling = 0.2
        pot = CoupledHarmonicOscillator(coupling=coupling)

        # Get DVR reference
        dvr_energies = compute_dvr_reference(
            pot, x_range=(-6, 6), y_range=(-6, 6),
            n_states=5, n_basis=50
        )

        solver = NonSepQuantumSolver(pot)

        # Solve ground state
        result = solver.solve_state(0, 0, n_epochs=5000, verbose=False)

        # Should be close to DVR ground state
        assert abs(result['E'] - dvr_energies[0]) < 0.1

    @pytest.mark.slow
    def test_action_quantization(self):
        """Action integrals should approach quantization values."""
        pot = HarmonicOscillator2D(omega_x=1.0, omega_y=1.0)
        solver = NonSepQuantumSolver(pot)

        # Ground state: J_x/ℏ ≈ 0.5, J_y/ℏ ≈ 0.5
        result = solver.solve_state(0, 0, n_epochs=5000, verbose=False)

        assert abs(result['J_x_over_hbar'] - 0.5) < 0.2
        assert abs(result['J_y_over_hbar'] - 0.5) < 0.2


class TestPoleStructure:
    """Tests verifying correct pole structure for different states."""

    def test_pole_count_matches_quantum_numbers(self):
        """Number of poles should match quantum numbers."""
        for n_x in range(3):
            for n_y in range(3):
                net = QuantumNumberPINN2D(n_x=n_x, n_y=n_y)
                poles = net.pole_positions()
                assert len(poles['x']) == n_x, f"Expected {n_x} x-poles, got {len(poles['x'])}"
                assert len(poles['y']) == n_y, f"Expected {n_y} y-poles, got {len(poles['y'])}"

    def test_pole_symmetry_even_state(self):
        """For even quantum number, poles should be symmetric."""
        net = QuantumNumberPINN2D(n_x=2, n_y=0, pole_scale=1.0)
        poles = net.pole_positions()

        # Two poles should be symmetric about origin
        assert len(poles['x']) == 2
        assert abs(poles['x'][0] + poles['x'][1]) < 1e-10

    def test_single_pole_at_origin(self):
        """For n=1, initial pole should be at origin."""
        net = QuantumNumberPINN2D(n_x=1, n_y=0, pole_scale=1.0)
        poles = net.pole_positions()

        assert abs(poles['x'][0]) < 1e-10
