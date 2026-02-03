"""
Tests for NNQS baseline implementation.

Validates:
1. Network architecture
2. Local energy computation
3. MCMC sampling
4. Training convergence
"""

import pytest
import numpy as np
import torch

from nnqs_baseline import (
    LogAmplitudeNetwork,
    MetropolisSampler,
    compute_local_energy,
    NNQSTrainer,
)
from nnqs_baseline.local_energy import ExactHarmonicOscillatorModel
from nnqs_baseline.train import harmonic_potential_1d, harmonic_potential_nd


class TestLogAmplitudeNetwork:
    """Test neural network architecture."""

    def test_network_creation(self):
        """Test network can be created."""
        net = LogAmplitudeNetwork(input_dim=1, hidden_dims=[64, 64])
        assert net.input_dim == 1
        assert net.hidden_dims == [64, 64]

    def test_forward_1d(self):
        """Test forward pass for 1D input."""
        net = LogAmplitudeNetwork(input_dim=1, hidden_dims=[32, 32])
        x = torch.randn(100)  # 1D input
        f = net(x)
        assert f.shape == (100,)

    def test_forward_nd(self):
        """Test forward pass for N-D input."""
        net = LogAmplitudeNetwork(input_dim=6, hidden_dims=[64, 64])
        x = torch.randn(100, 6)
        f = net(x)
        assert f.shape == (100,)

    def test_log_prob(self):
        """Test log_prob returns 2*f."""
        net = LogAmplitudeNetwork(input_dim=1)
        x = torch.randn(50)
        f = net(x)
        log_prob = net.log_prob(x)
        torch.testing.assert_close(log_prob, 2.0 * f)

    def test_initialization(self):
        """Test final layer is initialized to near-zero."""
        net = LogAmplitudeNetwork(input_dim=1, hidden_dims=[32, 32])
        # Initial f should be near 0 for any input
        x = torch.randn(100)
        f = net(x)
        assert torch.abs(f).max() < 1.0  # Should be small initially

    def test_get_momentum(self):
        """Test momentum extraction."""
        net = LogAmplitudeNetwork(input_dim=2, hidden_dims=[32, 32])
        x = torch.randn(50, 2, requires_grad=True)
        p_real, p_imag = net.get_momentum(x)
        assert p_real.shape == (50, 2)
        assert p_imag.shape == (50, 2)
        # For real wavefunction, p_real should be 0
        torch.testing.assert_close(p_real, torch.zeros_like(p_real))


class TestLocalEnergy:
    """Test local energy computation."""

    def test_exact_ground_state_1d(self):
        """Test E_loc = E_exact everywhere for exact ground state."""
        # For exact psi = exp(-x^2/2), E_loc should be 0.5 everywhere
        model = ExactHarmonicOscillatorModel(dim=1)

        x = torch.randn(100, 1)
        E_loc = compute_local_energy(model, x, harmonic_potential_1d)

        # All local energies should be 0.5
        assert torch.allclose(E_loc, torch.full_like(E_loc, 0.5), atol=1e-5)

    def test_exact_ground_state_6d(self):
        """Test E_loc = E_exact for 6D harmonic oscillator."""
        model = ExactHarmonicOscillatorModel(dim=6)

        x = torch.randn(100, 6)
        E_loc = compute_local_energy(model, x, harmonic_potential_nd)

        # E_exact = 3.0 for 6D harmonic oscillator
        assert torch.allclose(E_loc, torch.full_like(E_loc, 3.0), atol=1e-5)

    def test_local_energy_shape(self):
        """Test output shape of local energy."""
        net = LogAmplitudeNetwork(input_dim=2, hidden_dims=[32])
        x = torch.randn(100, 2)

        def V(x):
            return 0.5 * torch.sum(x**2, dim=-1)

        E_loc = compute_local_energy(net, x, V)
        assert E_loc.shape == (100,)


class TestMetropolisSampler:
    """Test MCMC sampler."""

    def test_sampler_creation(self):
        """Test sampler can be created."""
        net = LogAmplitudeNetwork(input_dim=1)
        sampler = MetropolisSampler(net, dim=1, n_walkers=100)
        assert sampler.n_walkers == 100
        assert sampler.dim == 1

    def test_sample_shape(self):
        """Test sample output shape."""
        net = LogAmplitudeNetwork(input_dim=2)
        sampler = MetropolisSampler(net, dim=2, n_walkers=500)
        samples = sampler.sample(n_steps=10)
        assert samples.shape == (500, 2)

    def test_acceptance_rate_computed(self):
        """Test acceptance rate is computed correctly."""
        net = LogAmplitudeNetwork(input_dim=1)
        sampler = MetropolisSampler(net, dim=1, n_walkers=1000, step_size=0.5)
        sampler.sample(n_steps=100)
        # Acceptance rate should be a valid probability
        assert 0.0 <= sampler.acceptance_rate <= 1.0
        # For initialized (near-uniform) network, acceptance should be high
        assert sampler.acceptance_rate > 0.5

    def test_sampling_from_known_distribution(self):
        """Test that samples from exact Gaussian match expected statistics."""
        # Use exact harmonic oscillator model
        model = ExactHarmonicOscillatorModel(dim=1)

        # Create a compatible sampler (need to wrap model)
        class ModelWrapper:
            def __init__(self, model):
                self.model = model
            def __call__(self, x):
                return self.model(x)
            def parameters(self):
                return iter([torch.zeros(1)])

        wrapper = ModelWrapper(model)
        sampler = MetropolisSampler(wrapper, dim=1, n_walkers=5000, step_size=0.5)
        sampler.thermalize(200)
        samples = sampler.sample(n_steps=100)

        # For ground state psi = exp(-x^2/2), |psi|^2 = exp(-x^2)
        # This is N(0, 0.5), so mean=0, var=0.5
        mean = samples.mean().item()
        var = samples.var().item()

        assert abs(mean) < 0.1, f"Mean {mean} should be near 0"
        assert abs(var - 0.5) < 0.1, f"Variance {var} should be near 0.5"


class TestNNQSTrainer:
    """Test the trainer class."""

    def test_trainer_creation(self):
        """Test trainer can be created."""
        trainer = NNQSTrainer(
            V_func=harmonic_potential_1d,
            dim=1,
            hidden_dims=[32, 32]
        )
        assert trainer.dim == 1

    def test_auto_scaling(self):
        """Test network auto-scaling with dimension."""
        trainer_1d = NNQSTrainer(harmonic_potential_1d, dim=1)
        trainer_6d = NNQSTrainer(harmonic_potential_nd, dim=6)

        # 6D should have larger network
        params_1d = sum(p.numel() for p in trainer_1d.model.parameters())
        params_6d = sum(p.numel() for p in trainer_6d.model.parameters())
        assert params_6d > params_1d


class TestTraining1D:
    """Test training on 1D harmonic oscillator."""

    @pytest.mark.slow
    def test_1d_convergence(self):
        """Test that 1D training converges to correct energy."""
        torch.manual_seed(42)

        trainer = NNQSTrainer(
            V_func=harmonic_potential_1d,
            dim=1,
            hidden_dims=[64, 64]
        )

        result = trainer.train(
            n_epochs=3000,
            n_walkers=500,
            lr=1e-3,
            verbose=False
        )

        E_final = result['final_energy']
        E_exact = 0.5

        rel_error = abs(E_final - E_exact) / E_exact
        assert rel_error < 0.05, f"Energy error {rel_error*100:.1f}% > 5%"

    @pytest.mark.slow
    def test_1d_energy_decreases(self):
        """Test that energy decreases during training."""
        torch.manual_seed(42)

        trainer = NNQSTrainer(
            V_func=harmonic_potential_1d,
            dim=1,
            hidden_dims=[32, 32]
        )

        result = trainer.train(
            n_epochs=1000,
            n_walkers=500,
            verbose=False
        )

        energies = result['energies']
        # Average of first 100 should be higher than average of last 100
        early_avg = np.mean(energies[:100])
        late_avg = np.mean(energies[-100:])
        assert late_avg < early_avg, "Energy should decrease during training"


# Marker configuration
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
