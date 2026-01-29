"""
Tests for the PINN-based Quantum Hamilton-Jacobi solver.
"""

import pytest
import numpy as np
import torch
from quantum_hj import HarmonicOscillator, QuantumHJSolver
from quantum_hj.pinn import ComplexMLP, MomentumPINN, QuantumHJPINN, PINNSolver


class TestComplexMLP:
    """Tests for the ComplexMLP neural network."""

    def test_output_shape(self):
        """Network outputs correct shape for batch input."""
        net = ComplexMLP(hidden_layers=2, hidden_size=32)
        x = torch.complex(torch.randn(10), torch.randn(10))
        p = net(x)
        assert p.shape == x.shape
        assert p.dtype == torch.complex64

    def test_single_input(self):
        """Network handles single complex input."""
        net = ComplexMLP()
        x = torch.complex(torch.tensor(1.0), torch.tensor(0.5))
        p = net(x)
        assert p.shape == x.shape

    def test_deterministic_output(self):
        """Same input gives same output (no stochasticity)."""
        net = ComplexMLP()
        net.eval()
        x = torch.complex(torch.tensor([1.0, 2.0]), torch.tensor([0.5, -0.5]))
        with torch.no_grad():
            p1 = net(x)
            p2 = net(x)
        torch.testing.assert_close(p1, p2)


class TestQuantumHJPINN:
    """Tests for the QuantumHJPINN trainer."""

    @pytest.fixture
    def pinn(self):
        """Create PINN for harmonic oscillator ground state."""
        potential = HarmonicOscillator()
        return QuantumHJPINN(potential, E=0.5, hbar=1.0, mass=1.0)

    def test_initialization(self, pinn):
        """PINN initializes correctly."""
        assert pinn.E == 0.5
        assert pinn.hbar == 1.0
        assert pinn.mass == 1.0
        assert isinstance(pinn.net, (ComplexMLP, MomentumPINN))

    def test_sample_collocation_points(self, pinn):
        """Collocation point sampling works correctly."""
        x = pinn.sample_collocation_points(100)
        assert x.shape == (100,)
        assert x.dtype == torch.complex64
        # Check range
        assert x.real.min() >= -5.0
        assert x.real.max() <= 5.0
        assert x.imag.min() >= -1.0
        assert x.imag.max() <= 1.0

    def test_physics_loss_computes(self, pinn):
        """Physics loss function runs without error."""
        x = pinn.sample_collocation_points(50)
        loss = pinn.physics_loss(x)
        assert loss.item() >= 0  # Loss should be non-negative
        assert torch.isfinite(loss)

    def test_training_reduces_loss(self, pinn):
        """Training reduces the physics loss."""
        # Get initial loss
        x = pinn.sample_collocation_points(100)
        initial_loss = pinn.physics_loss(x).item()

        # Train briefly
        pinn.train(n_epochs=200, lr=1e-2, n_collocation=100, verbose=False)

        # Check loss decreased
        final_loss = pinn.loss_history[-1]
        assert final_loss < initial_loss

    def test_evaluate_returns_complex(self, pinn):
        """Evaluate returns complex momentum values."""
        pinn.train(n_epochs=100, verbose=False)
        x = np.array([1.0 + 0.5j, 2.0 - 0.3j])
        p = pinn.evaluate(x)
        assert p.shape == x.shape
        assert np.iscomplexobj(p)


class TestPINNSolver:
    """Tests for the PINNSolver class."""

    @pytest.fixture
    def solver(self):
        """Create PINN solver for harmonic oscillator."""
        return PINNSolver(HarmonicOscillator())

    def test_contour_creation(self, solver):
        """Contour is created correctly."""
        contour = solver._create_contour(E=1.0)
        assert len(contour) == solver.n_points
        assert np.iscomplexobj(contour)

    def test_contour_encloses_turning_points(self, solver):
        """Contour extends beyond turning points."""
        E = 2.0
        contour = solver._create_contour(E)
        tp = solver.potential.turning_points(E)
        assert contour.real.min() < tp[0]
        assert contour.real.max() > tp[1]

    def test_train_for_energy(self, solver):
        """Can train PINN for specific energy."""
        pinn = solver.train_for_energy(E=0.5, n_epochs=200, verbose=False)
        assert isinstance(pinn, QuantumHJPINN)
        assert len(pinn.loss_history) > 0

    def test_compute_action(self, solver):
        """Action integral can be computed."""
        pinn = solver.train_for_energy(E=0.5, n_epochs=500, verbose=False)
        J = solver.compute_action(pinn, E=0.5)
        assert np.isfinite(J)


@pytest.mark.slow
class TestPINNAccuracy:
    """Tests for PINN accuracy learning the exact quantum momentum p = -ix."""

    @pytest.fixture
    def solver(self):
        """Create PINN solver for harmonic oscillator."""
        return PINNSolver(HarmonicOscillator())

    def test_ground_state_momentum_at_origin(self, solver):
        """p(0) should be approximately 0."""
        pinn = solver.train_for_energy(E=0.5, n_epochs=3000, verbose=False)
        p = pinn.evaluate(np.array([0.0 + 0j]))[0]
        assert np.abs(p) < 0.01, f"p(0) = {p}, expected ~0"

    def test_ground_state_momentum_at_one(self, solver):
        """p(1) should be approximately -i."""
        pinn = solver.train_for_energy(E=0.5, n_epochs=3000, verbose=False)
        p = pinn.evaluate(np.array([1.0 + 0j]))[0]
        p_exact = -1j
        assert np.abs(p - p_exact) < 0.01, f"p(1) = {p}, expected -i"

    def test_ground_state_momentum_complex(self, solver):
        """p(0.5 + 0.3j) should be approximately 0.3 - 0.5j."""
        pinn = solver.train_for_energy(E=0.5, n_epochs=3000, verbose=False)
        x = 0.5 + 0.3j
        p = pinn.evaluate(np.array([x]))[0]
        p_exact = -1j * x  # = 0.3 - 0.5j
        assert np.abs(p - p_exact) < 0.01, f"p({x}) = {p}, expected {p_exact}"


@pytest.mark.slow
class TestPINNPhysics:
    """Tests verifying the PINN satisfies the QHJ equation."""

    def test_qhj_equation_satisfied(self):
        """The learned p should satisfy p^2 + i*dp/dx = 2(E - V)."""
        potential = HarmonicOscillator()
        solver = PINNSolver(potential)
        pinn = solver.train_for_energy(E=0.5, n_epochs=3000, verbose=False)

        # Check at several points
        test_points = [0.0, 1.0, -1.0, 0.5]
        h = 1e-5

        for x in test_points:
            x_arr = np.array([x + 0j])
            p = pinn.evaluate(x_arr)[0]
            dp_dx = (pinn.evaluate(np.array([x + h + 0j]))[0] -
                     pinn.evaluate(np.array([x - h + 0j]))[0]) / (2 * h)

            V = 0.5 * x * x
            lhs = p**2 + 1j * dp_dx
            rhs = 2 * (0.5 - V)

            residual = abs(lhs - rhs)
            assert residual < 0.05, f"QHJ residual at x={x}: {residual}"
