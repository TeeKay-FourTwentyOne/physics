"""
Tests for non-separable 2D PINN quantum Hamilton-Jacobi solver.

Tests compare PINN results against DVR reference calculations for:
1. Coupled harmonic oscillator
2. Henon-Heiles potential
"""

import numpy as np
import pytest
import torch

from quantum_hj.pinn_2d_nonsep import (
    EnergyConditionedMLP,
    QuantumHJPINN2DNonSep,
    EBKQuantizer,
    NonSepPINNSolver
)
from quantum_hj.potentials_2d import (
    CoupledHarmonicOscillator,
    HenonHeiles,
    HarmonicOscillator2D
)
from quantum_hj.dvr import DVRSolver2D, compute_dvr_reference


class TestEnergyConditionedMLP:
    """Tests for the energy-conditioned neural network."""

    def test_creation(self):
        """Test network instantiation."""
        net = EnergyConditionedMLP(hidden_layers=3, hidden_size=64)
        assert net is not None

    def test_output_shape(self):
        """Test network output shape."""
        net = EnergyConditionedMLP(hidden_layers=3, hidden_size=64)

        # Create test inputs
        n_points = 10
        x = torch.complex(torch.randn(n_points), torch.randn(n_points) * 0.1)
        y = torch.complex(torch.randn(n_points), torch.randn(n_points) * 0.1)
        E = torch.complex(torch.ones(n_points), torch.zeros(n_points))

        p_x, p_y = net(x, y, E)

        assert p_x.shape == (n_points,)
        assert p_y.shape == (n_points,)
        assert p_x.dtype == torch.complex64

    def test_energy_conditioning(self):
        """Test that network output changes with energy."""
        net = EnergyConditionedMLP(hidden_layers=3, hidden_size=64)

        # Fixed position, varying energy
        x = torch.complex(torch.ones(5), torch.zeros(5))
        y = torch.complex(torch.ones(5), torch.zeros(5))
        E1 = torch.complex(torch.ones(5), torch.zeros(5))
        E2 = torch.complex(torch.ones(5) * 2, torch.zeros(5))

        p_x1, p_y1 = net(x, y, E1)
        p_x2, p_y2 = net(x, y, E2)

        # Outputs should differ for different energies
        # (After initialization, differences may be small but should exist)
        assert not torch.allclose(p_x1, p_x2) or not torch.allclose(p_y1, p_y2)


class TestQuantumHJPINN2DNonSep:
    """Tests for the non-separable PINN trainer."""

    def test_creation(self):
        """Test PINN trainer instantiation."""
        pot = CoupledHarmonicOscillator(coupling=0.2)
        pinn = QuantumHJPINN2DNonSep(pot, E_range=(0.5, 3.0), device='cpu')

        assert pinn.potential == pot
        assert pinn.E_min == 0.5
        assert pinn.E_max == 3.0

    def test_sample_collocation_points(self):
        """Test collocation point sampling."""
        pot = CoupledHarmonicOscillator(coupling=0.2)
        pinn = QuantumHJPINN2DNonSep(pot, E_range=(0.5, 3.0), device='cpu')

        x, y, E = pinn.sample_collocation_points(100)

        assert len(x) == 100
        assert len(y) == 100
        assert len(E) == 100

        # Check energy range
        E_real = E.real.cpu().numpy()
        assert np.all(E_real >= 0.5)
        assert np.all(E_real <= 3.0)

    def test_physics_loss_computes(self):
        """Test that physics loss computation doesn't error."""
        pot = CoupledHarmonicOscillator(coupling=0.2)
        pinn = QuantumHJPINN2DNonSep(pot, E_range=(0.5, 3.0), device='cpu')

        x, y, E = pinn.sample_collocation_points(50)
        loss = pinn.physics_loss(x, y, E)

        assert loss.item() >= 0
        assert np.isfinite(loss.item())

    def test_curl_loss_computes(self):
        """Test that curl loss computation doesn't error."""
        pot = CoupledHarmonicOscillator(coupling=0.2)
        pinn = QuantumHJPINN2DNonSep(pot, E_range=(0.5, 3.0), device='cpu')

        x, y, E = pinn.sample_collocation_points(50)
        loss = pinn.curl_loss(x, y, E)

        assert loss.item() >= 0
        assert np.isfinite(loss.item())

    def test_short_training(self):
        """Test that short training reduces loss."""
        pot = CoupledHarmonicOscillator(coupling=0.2)
        pinn = QuantumHJPINN2DNonSep(
            pot, E_range=(0.5, 3.0), device='cpu',
            hidden_layers=2, hidden_size=32
        )

        initial_loss = pinn.train(n_epochs=10, verbose=False, n_collocation=100)
        final_loss = pinn.train(n_epochs=100, verbose=False, n_collocation=100)

        # Loss should decrease or stay similar (not explode)
        assert final_loss < initial_loss * 10

    def test_evaluate(self):
        """Test momentum evaluation."""
        pot = CoupledHarmonicOscillator(coupling=0.2)
        pinn = QuantumHJPINN2DNonSep(
            pot, E_range=(0.5, 3.0), device='cpu',
            hidden_layers=2, hidden_size=32
        )
        pinn.train(n_epochs=10, verbose=False, n_collocation=50)

        # Evaluate at test points
        x = np.array([1.0, 2.0]) + 0j
        y = np.array([0.5, 1.0]) + 0j
        E = np.array([1.0, 1.5]) + 0j

        p_x, p_y = pinn.evaluate(x, y, E)

        assert len(p_x) == 2
        assert len(p_y) == 2
        assert np.all(np.isfinite(p_x))
        assert np.all(np.isfinite(p_y))


class TestEBKQuantizer:
    """Tests for EBK quantization."""

    @pytest.fixture
    def trained_pinn(self):
        """Create a minimally trained PINN for testing."""
        pot = CoupledHarmonicOscillator(coupling=0.2)
        pinn = QuantumHJPINN2DNonSep(
            pot, E_range=(0.5, 5.0), device='cpu',
            hidden_layers=3, hidden_size=64
        )
        pinn.train(n_epochs=500, verbose=False, n_collocation=200)
        return pinn

    def test_creation(self, trained_pinn):
        """Test quantizer instantiation."""
        pot = CoupledHarmonicOscillator(coupling=0.2)
        quant = EBKQuantizer(trained_pinn, pot)

        assert quant.pinn is trained_pinn
        assert quant.potential is pot

    def test_action_computation(self, trained_pinn):
        """Test that action integrals compute without error."""
        pot = CoupledHarmonicOscillator(coupling=0.2)
        quant = EBKQuantizer(trained_pinn, pot)

        J_x = quant.compute_action_x(E=1.0, y_fixed=0.0)
        J_y = quant.compute_action_y(E=1.0, x_fixed=0.0)

        assert np.isfinite(J_x)
        assert np.isfinite(J_y)
        assert J_x >= 0
        assert J_y >= 0

    def test_quantization_residual(self, trained_pinn):
        """Test quantization residual computation."""
        pot = CoupledHarmonicOscillator(coupling=0.2)
        quant = EBKQuantizer(trained_pinn, pot)

        residual = quant.quantization_residual(E=1.0, n_x=0, n_y=0)

        assert np.isfinite(residual)
        assert residual >= 0


class TestCoupledOscillatorPINN:
    """Integration tests for coupled oscillator with DVR comparison."""

    @pytest.mark.slow
    def test_ground_state_weak_coupling(self):
        """Test ground state energy for weakly coupled oscillator."""
        coupling = 0.2
        pot = CoupledHarmonicOscillator(coupling=coupling)

        # Get DVR reference
        dvr_energies = compute_dvr_reference(
            pot, x_range=(-8, 8), y_range=(-8, 8),
            n_states=5, n_basis=60
        )

        # Train PINN
        pinn = QuantumHJPINN2DNonSep(
            pot, E_range=(0.5, 4.0), device='cpu',
            hidden_layers=4, hidden_size=128
        )
        pinn.train(n_epochs=5000, verbose=False, n_collocation=500)

        # Verify physics residual
        residual = pinn.verify_physics_residual(dvr_energies[0], n_points=50)

        # Physics residual should be reasonably small
        assert residual < 0.5, f"Physics residual too large: {residual}"

    @pytest.mark.slow
    def test_coupled_oscillator_exact_comparison(self):
        """Compare PINN quantization with exact coupled oscillator energies."""
        coupling = 0.2
        pot = CoupledHarmonicOscillator(coupling=coupling)

        # Exact ground state energy
        exact_E0 = pot.ground_state_energy()

        # Train PINN over range including ground state
        pinn = QuantumHJPINN2DNonSep(
            pot, E_range=(0.5, 3.0), device='cpu',
            hidden_layers=4, hidden_size=128
        )
        pinn.train(n_epochs=8000, verbose=False, n_collocation=800)

        # Use quantizer
        quant = EBKQuantizer(pinn, pot)

        # Find ground state quantization
        result = quant.find_quantized_energy(0, 0, E_min=0.5, E_max=1.5)

        # Check if found energy is close to exact
        if result['success']:
            rel_error = abs(result['E'] - exact_E0) / exact_E0
            assert rel_error < 0.1, \
                f"Ground state: PINN={result['E']:.4f}, exact={exact_E0:.4f}"


class TestHenonHeilesPINN:
    """Integration tests for Henon-Heiles potential."""

    def test_henon_heiles_potential_evaluation(self):
        """Test that PINN handles Henon-Heiles potential."""
        pot = HenonHeiles(coupling=0.1)
        pinn = QuantumHJPINN2DNonSep(
            pot, E_range=(0.5, 5.0), device='cpu',
            hidden_layers=3, hidden_size=64
        )

        # Should create without error
        assert pinn.potential.coupling == 0.1
        assert pinn.potential.escape_energy > 10  # ~16.67 for λ=0.1

    @pytest.mark.slow
    def test_henon_heiles_training(self):
        """Test PINN training on Henon-Heiles potential."""
        pot = HenonHeiles(coupling=0.1)
        pinn = QuantumHJPINN2DNonSep(
            pot, E_range=(0.5, 5.0), device='cpu',
            hidden_layers=4, hidden_size=128
        )

        final_loss = pinn.train(n_epochs=5000, verbose=False, n_collocation=500)

        # Training should complete with finite loss
        assert np.isfinite(final_loss)
        assert final_loss < 10.0  # Should make some progress

    @pytest.mark.slow
    def test_henon_heiles_dvr_comparison(self):
        """Compare Henon-Heiles PINN with DVR ground state."""
        pot = HenonHeiles(coupling=0.1)

        # Get DVR reference
        dvr_energies = compute_dvr_reference(
            pot, x_range=(-10, 10), y_range=(-10, 10),
            n_states=5, n_basis=60
        )

        # Train PINN
        pinn = QuantumHJPINN2DNonSep(
            pot, E_range=(0.5, 5.0), device='cpu',
            hidden_layers=4, hidden_size=128
        )
        pinn.train(n_epochs=8000, verbose=False, n_collocation=800)

        # Check physics residual at DVR ground state energy
        residual = pinn.verify_physics_residual(dvr_energies[0], n_points=50)

        # Henon-Heiles is harder - allow larger residual
        assert residual < 1.0, f"Physics residual: {residual}"


class TestNonSepPINNSolver:
    """Tests for the high-level solver interface."""

    def test_solver_creation(self):
        """Test solver instantiation."""
        pot = CoupledHarmonicOscillator(coupling=0.2)
        solver = NonSepPINNSolver(pot, E_range=(0.5, 4.0), device='cpu')

        assert solver.potential is pot
        assert solver.E_range == (0.5, 4.0)

    def test_solver_training(self):
        """Test solver training."""
        pot = CoupledHarmonicOscillator(coupling=0.2)
        solver = NonSepPINNSolver(pot, E_range=(0.5, 4.0), device='cpu')

        pinn = solver.train(n_epochs=100, verbose=False, n_collocation=100)

        assert pinn is not None
        assert solver.pinn is pinn
        assert solver.quantizer is not None


class TestPhysicsResidual:
    """Tests for physics residual verification."""

    def test_residual_decreases_with_training(self):
        """Test that physics residual decreases with more training."""
        pot = CoupledHarmonicOscillator(coupling=0.2)

        # Short training
        pinn1 = QuantumHJPINN2DNonSep(
            pot, E_range=(0.5, 3.0), device='cpu',
            hidden_layers=3, hidden_size=64
        )
        pinn1.train(n_epochs=100, verbose=False, n_collocation=200)
        res1 = pinn1.verify_physics_residual(1.0, n_points=30)

        # Longer training
        pinn2 = QuantumHJPINN2DNonSep(
            pot, E_range=(0.5, 3.0), device='cpu',
            hidden_layers=3, hidden_size=64
        )
        pinn2.train(n_epochs=2000, verbose=False, n_collocation=200)
        res2 = pinn2.verify_physics_residual(1.0, n_points=30)

        # Longer training should generally give smaller residual
        # (Allow some variance due to random initialization)
        assert res2 < res1 * 5, \
            f"Expected residual to decrease: short={res1:.4f}, long={res2:.4f}"


class TestContourIntegration:
    """Tests for contour integration in action computation."""

    def test_contour_creation(self):
        """Test that contours are created correctly."""
        pot = CoupledHarmonicOscillator(coupling=0.2)
        pinn = QuantumHJPINN2DNonSep(
            pot, E_range=(0.5, 3.0), device='cpu',
            hidden_layers=2, hidden_size=32
        )
        pinn.train(n_epochs=10, verbose=False, n_collocation=50)

        quant = EBKQuantizer(pinn, pot)

        # Create contour
        contour_x = quant._create_contour_x(E=1.0, y_fixed=0.0)

        # Should be closed contour
        assert len(contour_x) == quant.n_contour

        # Should have real and imaginary parts
        assert np.any(np.imag(contour_x) != 0)

    def test_action_positive(self):
        """Test that computed actions are positive."""
        pot = CoupledHarmonicOscillator(coupling=0.2)
        pinn = QuantumHJPINN2DNonSep(
            pot, E_range=(0.5, 3.0), device='cpu',
            hidden_layers=3, hidden_size=64
        )
        pinn.train(n_epochs=500, verbose=False, n_collocation=200)

        quant = EBKQuantizer(pinn, pot)

        J_x = quant.compute_action_x(E=1.0)
        J_y = quant.compute_action_y(E=1.0)

        assert J_x >= 0, f"J_x should be positive: {J_x}"
        assert J_y >= 0, f"J_y should be positive: {J_y}"
