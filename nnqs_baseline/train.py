"""
Variational Monte Carlo training for Neural Network Quantum States.

The training objective is to minimize the variational energy:
    E[theta] = <psi_theta|H|psi_theta> / <psi_theta|psi_theta>
             = E_{|psi|^2}[E_loc]

The gradient uses the REINFORCE estimator (log-derivative trick):
    grad E = 2 * E_{|psi|^2}[(E_loc - E) * grad(f)]

This is equivalent to the covariance Cov(E_loc, f).
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Dict, List, Tuple

from .networks import LogAmplitudeNetwork
from .samplers import MetropolisSampler
from .local_energy import compute_local_energy


def train_nnqs(
    model: nn.Module,
    V_func: Callable,
    dim: int,
    n_epochs: int = 10000,
    n_walkers: int = 1000,
    lr: float = 1e-3,
    mcmc_steps: int = 10,
    burn_in: int = 100,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    log_interval: int = 100,
    hbar: float = 1.0,
    mass: float = 1.0,
    grad_clip: float = 1.0,
    max_sample_radius: float = 10.0,
) -> Dict:
    """
    Train NNQS using variational Monte Carlo.

    Args:
        model: LogAmplitudeNetwork
        V_func: Potential function V(x) -> energy
        dim: Number of spatial dimensions
        n_epochs: Number of training epochs
        n_walkers: Number of MCMC walkers
        lr: Learning rate
        mcmc_steps: MCMC steps between gradient updates
        burn_in: Initial burn-in steps for MCMC
        device: Torch device
        verbose: Print progress
        log_interval: Epochs between progress prints
        hbar: Reduced Planck constant
        mass: Particle mass
        grad_clip: Gradient clipping value
        max_sample_radius: Reset walkers that drift beyond this radius

    Returns:
        Dictionary with training history
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize sampler
    sampler = MetropolisSampler(model, dim, n_walkers, device=device)
    sampler.thermalize(burn_in)

    # Training history
    energies = []
    energy_stds = []
    acceptance_rates = []

    for epoch in range(n_epochs):
        # Sample from |psi|^2
        x = sampler.sample(n_steps=mcmc_steps)

        # Reset walkers that have drifted too far (stability measure)
        sample_radius = torch.sqrt(torch.sum(x ** 2, dim=-1))
        too_far = sample_radius > max_sample_radius
        if too_far.any():
            # Reset drifted walkers to random positions near origin
            n_reset = too_far.sum().item()
            sampler.walkers[too_far] = torch.randn(n_reset, dim, device=device)
            x = sampler.walkers.clone()

        x = x.requires_grad_(True)

        # Compute local energies
        E_loc = compute_local_energy(model, x, V_func, hbar=hbar, mass=mass)

        # Compute f = log|psi| for gradient estimator
        f = model(x)

        # Mean energy and standard deviation
        E_mean = E_loc.mean()
        E_std = E_loc.std()

        # REINFORCE gradient: loss = 2 * E[(E_loc - E_mean) * f]
        # We detach (E_loc - E_mean) to treat it as a constant weight
        centered_E = (E_loc - E_mean).detach()
        loss = 2.0 * torch.mean(centered_E * f)

        # Gradient step with clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Record history
        energies.append(E_mean.item())
        energy_stds.append(E_std.item())
        acceptance_rates.append(sampler.acceptance_rate)

        if verbose and epoch % log_interval == 0:
            print(f"Epoch {epoch:5d}: E = {E_mean.item():.6f} +/- {E_std.item():.6f}, "
                  f"accept = {sampler.acceptance_rate:.2f}")

    return {
        'energies': energies,
        'energy_stds': energy_stds,
        'acceptance_rates': acceptance_rates,
        'final_energy': energies[-1],
        'final_energy_std': energy_stds[-1],
    }


class NNQSTrainer:
    """
    Complete NNQS trainer with additional features.

    Features:
    - Automatic network sizing based on dimension
    - Energy variance monitoring for convergence
    - Checkpoint saving
    - Visualization
    - Gaussian pretraining for stable initialization
    """

    def __init__(
        self,
        V_func: Callable,
        dim: int,
        hidden_dims: Optional[List[int]] = None,
        hbar: float = 1.0,
        mass: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize trainer.

        Args:
            V_func: Potential function
            dim: Number of spatial dimensions
            hidden_dims: Network hidden layer sizes (auto-scaled if None)
            hbar: Reduced Planck constant
            mass: Particle mass
            device: Torch device
        """
        self.V_func = V_func
        self.dim = dim
        self.hbar = hbar
        self.mass = mass

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device

        # Auto-scale network based on dimension
        if hidden_dims is None:
            if dim == 1:
                hidden_dims = [64, 64]
            elif dim <= 3:
                hidden_dims = [128, 128]
            elif dim <= 6:
                hidden_dims = [256, 256, 256, 256]
            else:
                hidden_dims = [512] * 8

        self.model = LogAmplitudeNetwork(dim, hidden_dims).to(device)
        self.sampler = None

        # Training state
        self.energy_history = []
        self.std_history = []
        self.best_energy = float('inf')

    def pretrain_gaussian(self, n_epochs: int = 500, alpha: float = 0.5,
                          lr: float = 1e-3, verbose: bool = False):
        """
        Pretrain network to approximate Gaussian: f(x) = -alpha * |x|^2.

        This provides a much better starting point than uniform (f=0).
        For harmonic oscillator, alpha=0.5 gives the exact ground state.

        Args:
            n_epochs: Number of pretraining epochs
            alpha: Gaussian width parameter (0.5 for harmonic oscillator)
            lr: Learning rate
            verbose: Print progress
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(n_epochs):
            # Sample training points
            x = torch.randn(1000, self.dim, device=self.device) * 2.0

            # Target: f(x) = -alpha * |x|^2
            target = -alpha * torch.sum(x ** 2, dim=-1)

            # Prediction
            pred = self.model(x)

            # MSE loss
            loss = torch.mean((pred - target) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and epoch % 100 == 0:
                print(f"Pretrain {epoch}: loss = {loss.item():.6f}")

    def train(
        self,
        n_epochs: int = 10000,
        n_walkers: int = 1000,
        lr: float = 1e-3,
        mcmc_steps: int = 10,
        burn_in: int = 100,
        verbose: bool = True,
        log_interval: int = 100,
        convergence_window: int = 100,
        convergence_tol: float = 1e-4,
        checkpoint_path: Optional[str] = None,
        checkpoint_interval: int = 1000,
        pretrain: bool = True,
        pretrain_epochs: int = 500,
        grad_clip: float = 1.0,
        max_sample_radius: float = 10.0,
    ) -> Dict:
        """
        Train the NNQS model.

        Args:
            n_epochs: Maximum epochs
            n_walkers: Number of MCMC walkers
            lr: Learning rate
            mcmc_steps: MCMC steps per gradient update
            burn_in: Initial MCMC burn-in
            verbose: Print progress
            log_interval: Print interval
            convergence_window: Window size for convergence check
            convergence_tol: Tolerance for convergence
            checkpoint_path: Path to save checkpoints
            checkpoint_interval: Epochs between checkpoints
            pretrain: Whether to pretrain with Gaussian approximation
            pretrain_epochs: Number of pretraining epochs
            grad_clip: Gradient clipping value
            max_sample_radius: Reset walkers beyond this radius

        Returns:
            Training results dictionary
        """
        # Pretrain to Gaussian for stable initialization
        if pretrain:
            if verbose:
                print("Pretraining to Gaussian approximation...")
            self.pretrain_gaussian(n_epochs=pretrain_epochs, verbose=verbose)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Initialize sampler
        self.sampler = MetropolisSampler(
            self.model, self.dim, n_walkers, device=self.device
        )
        self.sampler.thermalize(burn_in)

        for epoch in range(n_epochs):
            # Sample
            x = self.sampler.sample(n_steps=mcmc_steps)

            # Reset walkers that have drifted too far
            sample_radius = torch.sqrt(torch.sum(x ** 2, dim=-1))
            too_far = sample_radius > max_sample_radius
            if too_far.any():
                n_reset = too_far.sum().item()
                self.sampler.walkers[too_far] = torch.randn(
                    n_reset, self.dim, device=self.device
                )
                x = self.sampler.walkers.clone()

            x = x.requires_grad_(True)

            # Compute local energy
            E_loc = compute_local_energy(
                self.model, x, self.V_func, self.hbar, self.mass
            )

            # Forward pass for gradient
            f = self.model(x)

            # Statistics
            E_mean = E_loc.mean()
            E_std = E_loc.std()

            # REINFORCE loss
            centered_E = (E_loc - E_mean).detach()
            loss = 2.0 * torch.mean(centered_E * f)

            # Update with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            optimizer.step()

            # Record
            self.energy_history.append(E_mean.item())
            self.std_history.append(E_std.item())

            if E_mean.item() < self.best_energy:
                self.best_energy = E_mean.item()

            # Progress
            if verbose and epoch % log_interval == 0:
                print(f"Epoch {epoch:5d}: E = {E_mean.item():.6f} +/- {E_std.item():.6f}")

            # Checkpoint
            if checkpoint_path and epoch > 0 and epoch % checkpoint_interval == 0:
                self.save_checkpoint(checkpoint_path)

            # Convergence check
            if epoch > convergence_window:
                recent = self.energy_history[-convergence_window:]
                if np.std(recent) < convergence_tol:
                    if verbose:
                        print(f"Converged at epoch {epoch}")
                    break

        return {
            'energies': self.energy_history,
            'energy_stds': self.std_history,
            'final_energy': self.energy_history[-1],
            'best_energy': self.best_energy,
            'n_epochs': len(self.energy_history),
        }

    def evaluate_energy(self, n_samples: int = 10000) -> Tuple[float, float]:
        """
        Evaluate energy with many samples for accurate estimate.

        Returns:
            (E_mean, E_std_error)  - mean and standard error of mean
        """
        if self.sampler is None:
            self.sampler = MetropolisSampler(
                self.model, self.dim, n_samples, device=self.device
            )
            self.sampler.thermalize(200)
        else:
            # Reset with more walkers
            self.sampler.n_walkers = n_samples
            self.sampler.walkers = torch.randn(
                n_samples, self.dim, device=self.device
            )
            self.sampler.thermalize(200)

        x = self.sampler.sample(n_steps=50)
        x = x.requires_grad_(True)

        # Note: compute_local_energy needs gradients enabled
        E_loc = compute_local_energy(
            self.model, x, self.V_func, self.hbar, self.mass
        )

        E_mean = E_loc.mean().item()
        E_std_error = E_loc.std().item() / np.sqrt(n_samples)

        return E_mean, E_std_error

    def get_samples(self, n_samples: int = 1000) -> np.ndarray:
        """Get samples from |psi|^2 distribution."""
        if self.sampler is None:
            self.sampler = MetropolisSampler(
                self.model, self.dim, n_samples, device=self.device
            )
            self.sampler.thermalize(100)

        x = self.sampler.sample(n_steps=50)
        return x.cpu().numpy()

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'dim': self.dim,
            'energy_history': self.energy_history,
            'std_history': self.std_history,
            'best_energy': self.best_energy,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.energy_history = checkpoint.get('energy_history', [])
        self.std_history = checkpoint.get('std_history', [])
        self.best_energy = checkpoint.get('best_energy', float('inf'))

    def plot_convergence(self, exact_energy: Optional[float] = None,
                         save_path: Optional[str] = None):
        """Plot energy convergence."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        epochs = np.arange(len(self.energy_history))

        # Energy vs epoch
        ax1.plot(epochs, self.energy_history, 'b-', alpha=0.7, label='Energy')
        if exact_energy is not None:
            ax1.axhline(exact_energy, color='r', linestyle='--',
                       label=f'Exact: {exact_energy:.6f}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Energy')
        ax1.set_title('NNQS Energy Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Energy standard deviation
        ax2.semilogy(epochs, self.std_history, 'g-', alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Energy Std Dev (log scale)')
        ax2.set_title('Local Energy Variance')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        return fig

    def plot_wavefunction_1d(self, x_range: Tuple[float, float] = (-4, 4),
                             n_points: int = 200,
                             exact_psi: Optional[Callable] = None,
                             save_path: Optional[str] = None):
        """
        Plot 1D wavefunction (only for dim=1).
        """
        if self.dim != 1:
            raise ValueError("1D plot only available for dim=1")

        x = torch.linspace(x_range[0], x_range[1], n_points, device=self.device)
        x = x.unsqueeze(-1)  # Shape: (n_points, 1)

        with torch.no_grad():
            f = self.model(x)
            psi_sq = torch.exp(2 * f)  # |psi|^2
            psi_sq = psi_sq / psi_sq.sum() * n_points / (x_range[1] - x_range[0])

        x_np = x.cpu().numpy().flatten()
        psi_sq_np = psi_sq.cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(x_np, psi_sq_np, 'b-', linewidth=2, label='NNQS |psi|^2')

        if exact_psi is not None:
            exact_sq = exact_psi(x_np) ** 2
            exact_sq = exact_sq / exact_sq.sum() * n_points / (x_range[1] - x_range[0])
            ax.plot(x_np, exact_sq, 'r--', linewidth=2, label='Exact |psi|^2')

        ax.set_xlabel('x')
        ax.set_ylabel('|psi(x)|^2')
        ax.set_title('Wavefunction Probability Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    def plot_samples_histogram(self, n_samples: int = 10000,
                               exact_dist: Optional[Callable] = None,
                               save_path: Optional[str] = None):
        """Plot histogram of MCMC samples vs exact distribution."""
        samples = self.get_samples(n_samples)

        if self.dim == 1:
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.hist(samples.flatten(), bins=50, density=True, alpha=0.7,
                   label='MCMC samples')

            if exact_dist is not None:
                x = np.linspace(samples.min(), samples.max(), 200)
                ax.plot(x, exact_dist(x), 'r-', linewidth=2, label='Exact |psi|^2')

            ax.set_xlabel('x')
            ax.set_ylabel('Probability density')
            ax.set_title('MCMC Sample Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

        else:
            # Multi-dimensional: plot marginals
            n_dims_to_plot = min(self.dim, 4)
            fig, axes = plt.subplots(1, n_dims_to_plot, figsize=(4*n_dims_to_plot, 4))

            for i in range(n_dims_to_plot):
                ax = axes[i] if n_dims_to_plot > 1 else axes
                ax.hist(samples[:, i], bins=50, density=True, alpha=0.7)
                ax.set_xlabel(f'x_{i}')
                ax.set_ylabel('Density')
                ax.set_title(f'Marginal x_{i}')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


# Potential functions for common systems
def harmonic_potential_1d(x: torch.Tensor, omega: float = 1.0) -> torch.Tensor:
    """1D harmonic oscillator: V = 0.5 * omega^2 * x^2."""
    if x.dim() == 1:
        return 0.5 * omega ** 2 * x ** 2
    return 0.5 * omega ** 2 * x[:, 0] ** 2


def harmonic_potential_nd(x: torch.Tensor, omega: float = 1.0) -> torch.Tensor:
    """N-D isotropic harmonic oscillator: V = 0.5 * omega^2 * |x|^2."""
    return 0.5 * omega ** 2 * torch.sum(x ** 2, dim=-1)


def coupled_harmonic_6d(x: torch.Tensor, coupling: float = 0.15) -> torch.Tensor:
    """
    6D coupled harmonic oscillator with periodic nearest-neighbor coupling.

    V = 0.5 * sum(x_i^2) + lambda * sum(x_i * x_{i+1})  (periodic)
    """
    # Diagonal terms
    V = 0.5 * torch.sum(x ** 2, dim=-1)

    # Coupling terms (periodic)
    for i in range(6):
        j = (i + 1) % 6
        V = V + coupling * x[:, i] * x[:, j]

    return V


if __name__ == '__main__':
    # Quick test: 1D harmonic oscillator
    print("Testing 1D Harmonic Oscillator NNQS")
    print("=" * 50)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create trainer
    trainer = NNQSTrainer(
        V_func=harmonic_potential_1d,
        dim=1,
        hidden_dims=[64, 64],
        device=device
    )

    # Train
    result = trainer.train(
        n_epochs=5000,
        n_walkers=1000,
        lr=1e-3,
        verbose=True,
        log_interval=500
    )

    # Evaluate
    E_final, E_err = trainer.evaluate_energy()
    print(f"\nFinal energy: {E_final:.6f} +/- {E_err:.6f}")
    print(f"Exact energy: 0.5")
    print(f"Error: {abs(E_final - 0.5) / 0.5 * 100:.2f}%")
