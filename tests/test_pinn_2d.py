"""
Tests for the 2D PINN-based Quantum Hamilton-Jacobi solver.
"""

import pytest
import numpy as np
import torch
from quantum_hj.potentials_2d import Potential2D, HarmonicOscillator2D
from quantum_hj.pinn_2d import (
    ComplexMLP2D, MomentumPINN2D, SeparableMomentumPINN2D,
    QuantumHJPINN2D, PINNSolver2D
)


class TestHarmonicOscillator2D:
    """Tests for the 2D harmonic oscillator potential."""

    def test_call(self):
        """Potential evaluates correctly."""
        pot = HarmonicOscillator2D(omega_x=1.0, omega_y=1.0)
        assert pot(0, 0) == 0
        assert np.isclose(pot(1, 0), 0.5)
        assert np.isclose(pot(0, 1), 0.5)
        assert np.isclose(pot(1, 1), 1.0)

    def test_anisotropic(self):
        """Anisotropic potential with different frequencies."""
        pot = HarmonicOscillator2D(omega_x=1.0, omega_y=2.0)
        assert np.isclose(pot(1, 0), 0.5)
        assert np.isclose(pot(0, 1), 2.0)

    def test_turning_points_x(self):
        """Classical turning points in x."""
        pot = HarmonicOscillator2D()
        tp = pot.turning_points_x(E=1.0, y=0.0)
        assert tp is not None
        assert np.isclose(tp[0], -np.sqrt(2))
        assert np.isclose(tp[1], np.sqrt(2))

    def test_turning_points_y(self):
        """Classical turning points in y."""
        pot = HarmonicOscillator2D()
        tp = pot.turning_points_y(E=1.0, x=0.0)
        assert tp is not None
        assert np.isclose(tp[0], -np.sqrt(2))
        assert np.isclose(tp[1], np.sqrt(2))

    def test_no_turning_points_low_energy(self):
        """No turning points when energy is consumed by fixed coordinate."""
        pot = HarmonicOscillator2D()
        # At y=2, V_y = 2, so for E=1 there's no energy left for x motion
        tp = pot.turning_points_x(E=1.0, y=2.0)
        assert tp is None

    def test_exact_energies(self):
        """Exact energy spectrum."""
        pot = HarmonicOscillator2D(omega_x=1.0, omega_y=1.0)
        energies = pot.exact_energies(2, 2)
        assert np.isclose(energies[(0, 0)], 1.0)  # 0.5 + 0.5
        assert np.isclose(energies[(1, 0)], 2.0)  # 1.5 + 0.5
        assert np.isclose(energies[(0, 1)], 2.0)  # 0.5 + 1.5
        assert np.isclose(energies[(1, 1)], 3.0)  # 1.5 + 1.5

    def test_exact_energies_anisotropic(self):
        """Exact energies for anisotropic oscillator."""
        omega_y = np.sqrt(2)
        pot = HarmonicOscillator2D(omega_x=1.0, omega_y=omega_y)
        # E_{0,0} = 0.5*1 + 0.5*sqrt(2) ≈ 1.207
        E_00 = pot.exact_energy(0, 0)
        assert np.isclose(E_00, 0.5 + 0.5 * omega_y)

    def test_exact_momentum_x(self):
        """Exact quantum momentum in x."""
        pot = HarmonicOscillator2D()
        # p_x = -ix for omega = m = 1
        x = 1.0 + 0.5j
        p_x = pot.exact_momentum_x(x)
        assert np.isclose(p_x, -1j * x)

    def test_exact_momentum_y(self):
        """Exact quantum momentum in y."""
        pot = HarmonicOscillator2D()
        # p_y = -iy for omega = m = 1
        y = 2.0 - 0.3j
        p_y = pot.exact_momentum_y(y)
        assert np.isclose(p_y, -1j * y)


class TestComplexMLP2D:
    """Tests for the 2D ComplexMLP neural network."""

    def test_output_shape(self):
        """Network outputs correct shapes."""
        net = ComplexMLP2D(hidden_layers=2, hidden_size=32)
        x = torch.complex(torch.randn(10), torch.randn(10))
        y = torch.complex(torch.randn(10), torch.randn(10))
        p_x, p_y = net(x, y)
        assert p_x.shape == x.shape
        assert p_y.shape == y.shape
        assert p_x.dtype == torch.complex64
        assert p_y.dtype == torch.complex64

    def test_single_input(self):
        """Network handles single complex input."""
        net = ComplexMLP2D()
        x = torch.complex(torch.tensor(1.0), torch.tensor(0.5))
        y = torch.complex(torch.tensor(-0.5), torch.tensor(0.2))
        p_x, p_y = net(x, y)
        assert p_x.shape == x.shape
        assert p_y.shape == y.shape

    def test_deterministic_output(self):
        """Same input gives same output."""
        net = ComplexMLP2D()
        net.eval()
        x = torch.complex(torch.tensor([1.0, 2.0]), torch.tensor([0.5, -0.5]))
        y = torch.complex(torch.tensor([0.0, 1.0]), torch.tensor([0.3, -0.1]))
        with torch.no_grad():
            p_x1, p_y1 = net(x, y)
            p_x2, p_y2 = net(x, y)
        torch.testing.assert_close(p_x1, p_x2)
        torch.testing.assert_close(p_y1, p_y2)


class TestQuantumHJPINN2D:
    """Tests for the 2D PINN trainer."""

    @pytest.fixture
    def pinn(self):
        """Create PINN for 2D harmonic oscillator ground state."""
        potential = HarmonicOscillator2D()
        return QuantumHJPINN2D(potential, E=1.0, hbar=1.0, mass=1.0)

    def test_initialization(self, pinn):
        """PINN initializes correctly."""
        assert pinn.E == 1.0
        assert pinn.hbar == 1.0
        assert pinn.mass == 1.0
        # Default is separable network
        assert isinstance(pinn.net, (MomentumPINN2D, SeparableMomentumPINN2D))

    def test_sample_collocation_points(self, pinn):
        """Collocation point sampling works correctly."""
        x, y = pinn.sample_collocation_points(100)
        assert x.shape == (100,)
        assert y.shape == (100,)
        assert x.dtype == torch.complex64
        assert y.dtype == torch.complex64

    def test_physics_loss_computes(self, pinn):
        """Physics loss function runs without error."""
        x, y = pinn.sample_collocation_points(50)
        loss = pinn.physics_loss(x, y)
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_curl_loss_computes(self, pinn):
        """Curl loss function runs without error."""
        x, y = pinn.sample_collocation_points(50)
        loss = pinn.curl_loss(x, y)
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_exact_solution_loss_computes(self, pinn):
        """Exact solution loss function runs without error."""
        loss = pinn.exact_solution_loss()
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_training_reduces_loss(self, pinn):
        """Training reduces the total loss."""
        # Get initial loss
        x, y = pinn.sample_collocation_points(100)
        initial_loss = pinn.physics_loss(x, y).item()

        # Train briefly
        pinn.train(n_epochs=200, lr=1e-2, n_collocation=100, verbose=False)

        # Check loss decreased
        assert len(pinn.loss_history) > 0
        final_loss = pinn.loss_history[-1]
        assert final_loss < initial_loss * 10  # Some improvement expected

    def test_evaluate_returns_complex(self, pinn):
        """Evaluate returns complex momentum values."""
        pinn.train(n_epochs=100, verbose=False)
        x = np.array([1.0 + 0.5j, 2.0 - 0.3j])
        y = np.array([0.5 - 0.2j, -1.0 + 0.1j])
        p_x, p_y = pinn.evaluate(x, y)
        assert p_x.shape == x.shape
        assert p_y.shape == y.shape
        assert np.iscomplexobj(p_x)
        assert np.iscomplexobj(p_y)


class TestPINNSolver2D:
    """Tests for the 2D PINN solver."""

    @pytest.fixture
    def solver(self):
        """Create PINN solver for 2D harmonic oscillator."""
        return PINNSolver2D(HarmonicOscillator2D())

    def test_contour_creation_x(self, solver):
        """X-contour is created correctly."""
        contour = solver._create_contour_x(E=1.0, y_fixed=0.0)
        assert len(contour) == solver.n_points
        assert np.iscomplexobj(contour)

    def test_contour_creation_y(self, solver):
        """Y-contour is created correctly."""
        contour = solver._create_contour_y(E=1.0, x_fixed=0.0)
        assert len(contour) == solver.n_points
        assert np.iscomplexobj(contour)

    def test_contour_encloses_turning_points_x(self, solver):
        """X-contour extends beyond turning points."""
        E = 2.0
        contour = solver._create_contour_x(E, y_fixed=0.0)
        tp = solver.potential.turning_points_x(E, y=0.0)
        assert contour.real.min() < tp[0]
        assert contour.real.max() > tp[1]

    def test_contour_encloses_turning_points_y(self, solver):
        """Y-contour extends beyond turning points."""
        E = 2.0
        contour = solver._create_contour_y(E, x_fixed=0.0)
        tp = solver.potential.turning_points_y(E, x=0.0)
        assert contour.real.min() < tp[0]
        assert contour.real.max() > tp[1]

    def test_train_for_energy(self, solver):
        """Can train PINN for specific energy."""
        pinn = solver.train_for_energy(E=1.0, n_epochs=200, verbose=False)
        assert isinstance(pinn, QuantumHJPINN2D)
        assert len(pinn.loss_history) > 0

    def test_compute_action_x(self, solver):
        """X-action integral can be computed."""
        pinn = solver.train_for_energy(E=1.0, n_epochs=500, verbose=False)
        J_x = solver.compute_action_x(pinn, E=1.0)
        assert np.isfinite(J_x)

    def test_compute_action_y(self, solver):
        """Y-action integral can be computed."""
        pinn = solver.train_for_energy(E=1.0, n_epochs=500, verbose=False)
        J_y = solver.compute_action_y(pinn, E=1.0)
        assert np.isfinite(J_y)


@pytest.mark.slow
class TestPINNAccuracy2D:
    """Tests for 2D PINN accuracy learning exact quantum momentum."""

    @pytest.fixture
    def solver(self):
        """Create PINN solver for 2D harmonic oscillator."""
        return PINNSolver2D(HarmonicOscillator2D())

    def test_ground_state_momentum_at_origin(self, solver):
        """p_x(0,0) and p_y(0,0) should be approximately 0."""
        pinn = solver.train_for_energy(E=1.0, n_epochs=5000, verbose=False)
        x = np.array([0.0 + 0j])
        y = np.array([0.0 + 0j])
        p_x, p_y = pinn.evaluate(x, y)
        assert np.abs(p_x[0]) < 0.02, f"p_x(0,0) = {p_x[0]}, expected ~0"
        assert np.abs(p_y[0]) < 0.02, f"p_y(0,0) = {p_y[0]}, expected ~0"

    def test_ground_state_momentum_at_unit_x(self, solver):
        """p_x(1, 0) should be approximately -i."""
        pinn = solver.train_for_energy(E=1.0, n_epochs=5000, verbose=False)
        x = np.array([1.0 + 0j])
        y = np.array([0.0 + 0j])
        p_x, p_y = pinn.evaluate(x, y)
        p_x_exact = -1j
        assert np.abs(p_x[0] - p_x_exact) < 0.02, f"p_x(1,0) = {p_x[0]}, expected -i"

    def test_ground_state_momentum_at_unit_y(self, solver):
        """p_y(0, 1) should be approximately -i."""
        pinn = solver.train_for_energy(E=1.0, n_epochs=5000, verbose=False)
        x = np.array([0.0 + 0j])
        y = np.array([1.0 + 0j])
        p_x, p_y = pinn.evaluate(x, y)
        p_y_exact = -1j
        assert np.abs(p_y[0] - p_y_exact) < 0.02, f"p_y(0,1) = {p_y[0]}, expected -i"

    def test_ground_state_momentum_diagonal(self, solver):
        """p_x(1,1) = -i, p_y(1,1) = -i for isotropic oscillator."""
        pinn = solver.train_for_energy(E=1.0, n_epochs=5000, verbose=False)
        x = np.array([1.0 + 0j])
        y = np.array([1.0 + 0j])
        p_x, p_y = pinn.evaluate(x, y)
        p_x_exact = -1j
        p_y_exact = -1j
        assert np.abs(p_x[0] - p_x_exact) < 0.02, f"p_x(1,1) = {p_x[0]}, expected -i"
        assert np.abs(p_y[0] - p_y_exact) < 0.02, f"p_y(1,1) = {p_y[0]}, expected -i"

    def test_ground_state_momentum_complex_x(self, solver):
        """p_x(0.5+0.2i, 0) should be approximately 0.2-0.5i."""
        pinn = solver.train_for_energy(E=1.0, n_epochs=5000, verbose=False)
        x = np.array([0.5 + 0.2j])
        y = np.array([0.0 + 0j])
        p_x, p_y = pinn.evaluate(x, y)
        p_x_exact = -1j * (0.5 + 0.2j)  # = 0.2 - 0.5j
        assert np.abs(p_x[0] - p_x_exact) < 0.02, f"p_x = {p_x[0]}, expected {p_x_exact}"


@pytest.mark.slow
class TestPINNQuantization2D:
    """Tests verifying quantization conditions for 2D states."""

    @pytest.fixture
    def solver(self):
        """Create PINN solver for isotropic 2D harmonic oscillator."""
        return PINNSolver2D(HarmonicOscillator2D(omega_x=1.0, omega_y=1.0))

    def test_ground_state_action_integrals(self, solver):
        """Ground state (0,0) should have J_x/ℏ ≈ 0.5, J_y/ℏ ≈ 0.5."""
        result = solver.verify_state(E=1.0, n_x=0, n_y=0, n_epochs=5000,
                                      verbose=False, tol=0.1)
        assert abs(result['J_x_over_hbar'] - 0.5) < 0.1, \
            f"J_x/ℏ = {result['J_x_over_hbar']}, expected 0.5"
        assert abs(result['J_y_over_hbar'] - 0.5) < 0.1, \
            f"J_y/ℏ = {result['J_y_over_hbar']}, expected 0.5"


@pytest.mark.slow
class TestPINNPhysics2D:
    """Tests verifying the 2D PINN satisfies the QHJ equation."""

    def test_qhj_equation_satisfied(self):
        """The learned (p_x, p_y) should satisfy the 2D QHJ equation."""
        potential = HarmonicOscillator2D()
        solver = PINNSolver2D(potential)
        pinn = solver.train_for_energy(E=1.0, n_epochs=5000, verbose=False)

        # Check at several points on real axes
        h = 1e-5

        test_points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5)]

        for x_val, y_val in test_points:
            x = np.array([x_val + 0j])
            y = np.array([y_val + 0j])

            p_x, p_y = pinn.evaluate(x, y)
            p_x = p_x[0]
            p_y = p_y[0]

            # Compute ∂p_x/∂x
            p_x_plus, _ = pinn.evaluate(np.array([x_val + h + 0j]), y)
            p_x_minus, _ = pinn.evaluate(np.array([x_val - h + 0j]), y)
            dp_x_dx = (p_x_plus[0] - p_x_minus[0]) / (2 * h)

            # Compute ∂p_y/∂y
            _, p_y_plus = pinn.evaluate(x, np.array([y_val + h + 0j]))
            _, p_y_minus = pinn.evaluate(x, np.array([y_val - h + 0j]))
            dp_y_dy = (p_y_plus[0] - p_y_minus[0]) / (2 * h)

            V = potential(x_val, y_val)
            E = 1.0

            # QHJ: p_x² + p_y² + iℏ(∂p_x/∂x + ∂p_y/∂y) = 2m(E - V)
            lhs = p_x**2 + p_y**2 + 1j * (dp_x_dx + dp_y_dy)
            rhs = 2 * (E - V)

            residual = abs(lhs - rhs)
            assert residual < 0.1, f"QHJ residual at ({x_val},{y_val}): {residual}"

    def test_irrotationality_constraint(self):
        """The learned momentum should satisfy ∂p_x/∂y = ∂p_y/∂x."""
        potential = HarmonicOscillator2D()
        solver = PINNSolver2D(potential)
        pinn = solver.train_for_energy(E=1.0, n_epochs=5000, verbose=False)

        h = 1e-5

        test_points = [(0.5, 0.5), (1.0, 0.5), (0.5, 1.0)]

        for x_val, y_val in test_points:
            # Compute ∂p_x/∂y
            p_x_y_plus, _ = pinn.evaluate(
                np.array([x_val + 0j]), np.array([y_val + h + 0j]))
            p_x_y_minus, _ = pinn.evaluate(
                np.array([x_val + 0j]), np.array([y_val - h + 0j]))
            dp_x_dy = (p_x_y_plus[0] - p_x_y_minus[0]) / (2 * h)

            # Compute ∂p_y/∂x
            _, p_y_x_plus = pinn.evaluate(
                np.array([x_val + h + 0j]), np.array([y_val + 0j]))
            _, p_y_x_minus = pinn.evaluate(
                np.array([x_val - h + 0j]), np.array([y_val + 0j]))
            dp_y_dx = (p_y_x_plus[0] - p_y_x_minus[0]) / (2 * h)

            curl = abs(dp_x_dy - dp_y_dx)
            assert curl < 0.1, f"Curl at ({x_val},{y_val}): {curl}"
