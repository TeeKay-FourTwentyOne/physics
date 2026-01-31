"""
Tests for the N-dimensional quantum-number-conditioned PINN solver.

These tests validate the PINN approach for 3D and 4D quantum systems.
"""

import pytest
import numpy as np
import torch

from quantum_hj.potentials_nd import (
    PotentialND, IsotropicOscillatorND,
    CoupledOscillator3D, CoupledOscillator4D
)
from quantum_hj.pinn_nd import (
    hermite_zeros, QuantumNumberPINNND,
    QuantumNumberTrainerND, NonSepQuantumSolverND
)


class TestHermiteZeros:
    """Test the Hermite polynomial zeros computation."""

    def test_n_zero(self):
        """n=0 should return empty array."""
        zeros = hermite_zeros(0)
        assert len(zeros) == 0

    def test_n_one(self):
        """n=1 should return [0]."""
        zeros = hermite_zeros(1)
        assert len(zeros) == 1
        assert abs(zeros[0]) < 1e-10

    def test_n_two(self):
        """n=2 should return symmetric zeros around 0."""
        zeros = hermite_zeros(2)
        assert len(zeros) == 2
        assert abs(zeros[0] + zeros[1]) < 1e-10  # Symmetric

    def test_symmetry(self):
        """Hermite zeros should be symmetric around 0."""
        for n in range(1, 6):
            zeros = hermite_zeros(n)
            assert len(zeros) == n
            # Sum should be ~0 (symmetric)
            assert abs(np.sum(zeros)) < 1e-10


class TestQuantumNumberPINNND:
    """Test the N-D PINN network architecture."""

    def test_3d_network_creation(self):
        """Test creating a 3D network."""
        net = QuantumNumberPINNND((1, 0, 2), hidden_size=64)
        assert net.ndim == 3
        assert net.quantum_numbers == (1, 0, 2)
        assert len(net.poles) == 3
        assert len(net.poles[0]) == 1  # 1 pole in dim 0
        assert len(net.poles[1]) == 0  # 0 poles in dim 1
        assert len(net.poles[2]) == 2  # 2 poles in dim 2

    def test_4d_network_creation(self):
        """Test creating a 4D network."""
        net = QuantumNumberPINNND((0, 1, 1, 0), hidden_size=64)
        assert net.ndim == 4
        assert net.quantum_numbers == (0, 1, 1, 0)

    def test_3d_forward_pass(self):
        """Test forward pass with 3D input."""
        net = QuantumNumberPINNND((0, 0, 0), hidden_size=32)

        x = torch.complex(torch.randn(10), torch.randn(10))
        y = torch.complex(torch.randn(10), torch.randn(10))
        z = torch.complex(torch.randn(10), torch.randn(10))

        p_x, p_y, p_z = net(x, y, z)

        assert p_x.shape == (10,)
        assert p_y.shape == (10,)
        assert p_z.shape == (10,)
        assert p_x.dtype == torch.complex64

    def test_4d_forward_pass(self):
        """Test forward pass with 4D input."""
        net = QuantumNumberPINNND((0, 0, 0, 0), hidden_size=32)

        coords = [torch.complex(torch.randn(5), torch.randn(5)) for _ in range(4)]
        momenta = net(*coords)

        assert len(momenta) == 4
        for p in momenta:
            assert p.shape == (5,)

    def test_pole_positions(self):
        """Test that pole positions are returned correctly."""
        net = QuantumNumberPINNND((2, 0, 1), hidden_size=32)
        poles = net.pole_positions()

        assert len(poles) == 3
        assert len(poles[0]) == 2
        assert len(poles[1]) == 0
        assert len(poles[2]) == 1


class TestPotentials3D:
    """Test 3D potential classes."""

    def test_isotropic_3d_creation(self):
        """Test creating 3D isotropic oscillator."""
        pot = IsotropicOscillatorND(ndim=3, omega=1.0)
        assert pot.ndim == 3
        assert pot.omega == 1.0

    def test_isotropic_3d_evaluation(self):
        """Test potential evaluation."""
        pot = IsotropicOscillatorND(ndim=3, omega=1.0)
        V = pot(1.0, 2.0, 3.0)
        expected = 0.5 * (1**2 + 2**2 + 3**2)
        assert abs(V - expected) < 1e-10

    def test_isotropic_3d_exact_energy(self):
        """Test exact energy formula."""
        pot = IsotropicOscillatorND(ndim=3, omega=1.0)

        # Ground state: E = 3/2
        E = pot.exact_energy((0, 0, 0))
        assert abs(E - 1.5) < 1e-10

        # First excited: E = 5/2
        E = pot.exact_energy((1, 0, 0))
        assert abs(E - 2.5) < 1e-10

    def test_coupled_3d_creation(self):
        """Test creating 3D coupled oscillator."""
        pot = CoupledOscillator3D(lambda1=0.2, lambda2=0.15)
        assert pot.ndim == 3

    def test_coupled_3d_hessian(self):
        """Test Hessian matrix structure."""
        pot = CoupledOscillator3D(lambda1=0.2, lambda2=0.15)
        H = pot.hessian_at_minimum()

        assert H.shape == (3, 3)
        # Check structure
        assert abs(H[0, 0] - 1.0) < 1e-10
        assert abs(H[0, 1] - 0.2) < 1e-10
        assert abs(H[0, 2]) < 1e-10
        assert abs(H[1, 2] - 0.15) < 1e-10

    def test_coupled_3d_alpha_matrix(self):
        """Test alpha matrix computation."""
        pot = CoupledOscillator3D(lambda1=0.2, lambda2=0.15)
        alpha = pot.alpha_matrix()

        assert alpha.shape == (3, 3)
        # Should be symmetric
        assert np.allclose(alpha, alpha.T)


class TestPotentials4D:
    """Test 4D potential classes."""

    def test_isotropic_4d_creation(self):
        """Test creating 4D isotropic oscillator."""
        pot = IsotropicOscillatorND(ndim=4, omega=1.0)
        assert pot.ndim == 4

    def test_isotropic_4d_exact_energy(self):
        """Test exact energy formula."""
        pot = IsotropicOscillatorND(ndim=4, omega=1.0)

        # Ground state: E = 2.0
        E = pot.exact_energy((0, 0, 0, 0))
        assert abs(E - 2.0) < 1e-10

    def test_coupled_4d_creation(self):
        """Test creating 4D coupled oscillator."""
        pot = CoupledOscillator4D(coupling=0.2)
        assert pot.ndim == 4
        assert abs(pot.omega_plus - np.sqrt(1.2)) < 1e-10
        assert abs(pot.omega_minus - np.sqrt(0.8)) < 1e-10

    def test_coupled_4d_alpha_matrix(self):
        """Test alpha matrix block structure."""
        pot = CoupledOscillator4D(coupling=0.2)
        alpha = pot.alpha_matrix()

        assert alpha.shape == (4, 4)

        # Check block structure: [[A, 0], [0, A]] where A is 2x2
        assert abs(alpha[0, 2]) < 1e-10
        assert abs(alpha[0, 3]) < 1e-10
        assert abs(alpha[1, 2]) < 1e-10
        assert abs(alpha[1, 3]) < 1e-10

        # The two blocks should be identical
        assert np.allclose(alpha[:2, :2], alpha[2:, 2:])


class TestTrainerND:
    """Test the N-D trainer class."""

    def test_trainer_3d_creation(self):
        """Test creating a 3D trainer."""
        pot = IsotropicOscillatorND(ndim=3)
        trainer = QuantumNumberTrainerND(pot, (0, 0, 0))
        assert trainer.ndim == 3

    def test_trainer_4d_creation(self):
        """Test creating a 4D trainer."""
        pot = IsotropicOscillatorND(ndim=4)
        trainer = QuantumNumberTrainerND(pot, (0, 0, 0, 0))
        assert trainer.ndim == 4

    def test_sample_collocation_points_3d(self):
        """Test collocation point sampling in 3D."""
        pot = IsotropicOscillatorND(ndim=3)
        trainer = QuantumNumberTrainerND(pot, (0, 0, 0))

        coords = trainer.sample_collocation_points(100)
        assert len(coords) == 3
        for x in coords:
            assert x.shape == (100,)
            assert x.dtype == torch.complex64

    def test_physics_loss_computes(self):
        """Test that physics loss computes without error."""
        pot = IsotropicOscillatorND(ndim=3)
        trainer = QuantumNumberTrainerND(pot, (0, 0, 0), hidden_size=32)

        coords = trainer.sample_collocation_points(50)
        loss = trainer.physics_loss(*coords)

        assert isinstance(loss.item(), float)
        assert loss.item() >= 0

    def test_curl_loss_computes(self):
        """Test that curl loss computes without error."""
        pot = IsotropicOscillatorND(ndim=3)
        trainer = QuantumNumberTrainerND(pot, (0, 0, 0), hidden_size=32)

        coords = trainer.sample_collocation_points(50)
        loss = trainer.curl_loss(*coords)

        assert isinstance(loss.item(), float)
        assert loss.item() >= 0


@pytest.mark.slow
class TestTraining3D:
    """Integration tests for 3D PINN training."""

    def test_3d_isotropic_ground_state(self):
        """
        Test 3D isotropic oscillator ground state.

        Target: E = 1.5 with < 5% error (relaxed for faster tests)
        """
        pot = IsotropicOscillatorND(ndim=3, omega=1.0)
        trainer = QuantumNumberTrainerND(pot, (0, 0, 0), hidden_size=64)

        result = trainer.train(
            n_epochs=5000,
            n_collocation=500,
            quant_start_epoch=1000,
            verbose=False
        )

        exact_E = 1.5
        rel_error = abs(result['E'] - exact_E) / exact_E

        assert rel_error < 0.05, f"Ground state error {rel_error:.1%} > 5%"

    def test_3d_isotropic_first_excited(self):
        """
        Test 3D isotropic oscillator first excited state.

        Target: E = 2.5 with < 5% error (relaxed for faster tests)
        """
        pot = IsotropicOscillatorND(ndim=3, omega=1.0)
        trainer = QuantumNumberTrainerND(pot, (1, 0, 0), hidden_size=64)

        result = trainer.train(
            n_epochs=5000,
            n_collocation=500,
            quant_start_epoch=1000,
            verbose=False
        )

        exact_E = 2.5
        rel_error = abs(result['E'] - exact_E) / exact_E

        assert rel_error < 0.05, f"First excited error {rel_error:.1%} > 5%"

    def test_3d_coupled_ground_state(self):
        """
        Test 3D coupled oscillator ground state.

        Target: < 5% error compared to exact (relaxed for faster tests)
        """
        pot = CoupledOscillator3D(lambda1=0.2, lambda2=0.15)
        trainer = QuantumNumberTrainerND(pot, (0, 0, 0), hidden_size=64)

        result = trainer.train(
            n_epochs=6000,
            n_collocation=500,
            quant_start_epoch=1500,
            verbose=False
        )

        exact_E = pot.exact_energy((0, 0, 0))
        rel_error = abs(result['E'] - exact_E) / exact_E

        assert rel_error < 0.05, f"3D coupled ground state error {rel_error:.1%} > 5%"


@pytest.mark.slow
class TestTraining4D:
    """Integration tests for 4D PINN training."""

    def test_4d_isotropic_ground_state(self):
        """
        Test 4D isotropic oscillator ground state.

        Target: E = 2.0 with < 5% error (relaxed for faster tests)
        """
        pot = IsotropicOscillatorND(ndim=4, omega=1.0)
        trainer = QuantumNumberTrainerND(pot, (0, 0, 0, 0), hidden_size=64)

        result = trainer.train(
            n_epochs=5000,
            n_collocation=500,
            quant_start_epoch=1000,
            verbose=False
        )

        exact_E = 2.0
        rel_error = abs(result['E'] - exact_E) / exact_E

        assert rel_error < 0.05, f"4D ground state error {rel_error:.1%} > 5%"

    def test_4d_coupled_ground_state(self):
        """
        Test 4D coupled oscillator ground state.

        Target: < 5% error compared to exact (relaxed for faster tests)
        """
        pot = CoupledOscillator4D(coupling=0.2)
        trainer = QuantumNumberTrainerND(pot, (0, 0, 0, 0), hidden_size=64)

        result = trainer.train(
            n_epochs=6000,
            n_collocation=500,
            quant_start_epoch=1500,
            verbose=False
        )

        exact_E = pot.exact_energy((0, 0, 0, 0))
        rel_error = abs(result['E'] - exact_E) / exact_E

        assert rel_error < 0.05, f"4D coupled ground state error {rel_error:.1%} > 5%"


class TestNonSepQuantumSolverND:
    """Test the high-level solver class."""

    def test_solver_creation_3d(self):
        """Test creating a 3D solver."""
        pot = IsotropicOscillatorND(ndim=3)
        solver = NonSepQuantumSolverND(pot)
        assert solver.potential.ndim == 3

    def test_solver_creation_4d(self):
        """Test creating a 4D solver."""
        pot = IsotropicOscillatorND(ndim=4)
        solver = NonSepQuantumSolverND(pot)
        assert solver.potential.ndim == 4

    def test_exact_energy_access(self):
        """Test accessing exact energy through solver."""
        pot = IsotropicOscillatorND(ndim=3)
        solver = NonSepQuantumSolverND(pot)

        E = solver.exact_energy((0, 0, 0))
        assert abs(E - 1.5) < 1e-10


class TestMomentumStructure:
    """Test that momentum has correct structure."""

    def test_pole_contribution_3d(self):
        """Test that poles contribute correctly in 3D."""
        net = QuantumNumberPINNND((1, 0, 0), hidden_size=32)

        # Evaluate near the pole
        pole_pos = net.poles[0][0].item()

        # Slightly away from pole
        x = torch.complex(
            torch.tensor([pole_pos + 0.1]),
            torch.tensor([0.0])
        )
        y = torch.complex(torch.tensor([0.0]), torch.tensor([0.0]))
        z = torch.complex(torch.tensor([0.0]), torch.tensor([0.0]))

        p_x, p_y, p_z = net(x, y, z)

        # p_x should be large due to 1/(x - pole)
        # The pole contribution is 1/(0.1) = 10
        assert abs(p_x[0].real) > 5.0  # Pole dominates

    def test_no_poles_smooth_output(self):
        """Test that ground state (no poles) gives smooth output."""
        net = QuantumNumberPINNND((0, 0, 0), hidden_size=32)

        x = torch.complex(torch.linspace(-2, 2, 10), torch.zeros(10))
        y = torch.complex(torch.zeros(10), torch.zeros(10))
        z = torch.complex(torch.zeros(10), torch.zeros(10))

        p_x, p_y, p_z = net(x, y, z)

        # Should all be finite (no poles)
        assert torch.all(torch.isfinite(p_x.real))
        assert torch.all(torch.isfinite(p_x.imag))
