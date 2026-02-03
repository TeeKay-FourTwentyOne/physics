"""
NNQS Phase 2: 6D Separable Harmonic Oscillator

System:
    V(x) = 1/2 * sum(x_i^2)  for i = 1 to 6
    E_exact = 3.0  (ground state: 6 * 0.5 = 3.0)

This script trains the NNQS on the same 6D separable system
used for LP+PINN validation, enabling direct comparison.
"""

import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnqs_baseline import LogAmplitudeNetwork, MetropolisSampler, compute_local_energy
from nnqs_baseline.train import harmonic_potential_nd


def train_6d_separable():
    print("=" * 70)
    print("NNQS Phase 2: 6D Separable Harmonic Oscillator")
    print("V(x) = 0.5 * sum(x_i^2), E_exact = 3.0")
    print("=" * 70)

    # Configuration
    E_EXACT = 3.0
    MAX_WALL_TIME = 2 * 3600  # 2 hours in seconds
    TARGET_ERROR = 0.01  # 1%

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    # Network: 6D input, larger hidden layers
    hidden_dims = [512, 512, 512, 512]
    model = LogAmplitudeNetwork(input_dim=6, hidden_dims=hidden_dims).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Network: {hidden_dims}")
    print(f"Parameters: {n_params:,}")

    # Sampler settings
    n_walkers = 10000
    step_size = 0.5
    mcmc_steps = 20

    sampler = MetropolisSampler(model, dim=6, n_walkers=n_walkers,
                                 step_size=step_size, device=device)

    # Training settings
    lr = 5e-4
    grad_clip = 1.0
    max_epochs = 50000
    max_sample_radius = 15.0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining config:")
    print(f"  Learning rate: {lr}")
    print(f"  Walkers: {n_walkers}")
    print(f"  MCMC steps/epoch: {mcmc_steps}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Max wall time: {MAX_WALL_TIME/3600:.1f} hours")

    # ========================================
    # Phase 1: Pretrain to Gaussian
    # ========================================
    print("\n" + "-" * 70)
    print("Phase 1: Pretraining to Gaussian approximation")
    print("-" * 70)

    pretrain_epochs = 1000
    pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(pretrain_epochs):
        # Sample training points
        x = torch.randn(2000, 6, device=device) * 2.0

        # Target: f(x) = -0.5 * |x|^2 (exact ground state)
        target = -0.5 * torch.sum(x ** 2, dim=-1)

        # Prediction
        pred = model(x)

        # MSE loss
        loss = torch.mean((pred - target) ** 2)

        pretrain_optimizer.zero_grad()
        loss.backward()
        pretrain_optimizer.step()

        if epoch % 200 == 0:
            print(f"  Pretrain epoch {epoch}: loss = {loss.item():.6f}")

    print("Pretraining complete.")

    # ========================================
    # Phase 2: VMC Training
    # ========================================
    print("\n" + "-" * 70)
    print("Phase 2: VMC Training")
    print("-" * 70)

    # Thermalize sampler after pretraining
    sampler.thermalize(200)

    # Training history
    energies = []
    energy_stds = []
    acceptance_rates = []

    start_time = time.time()
    epoch_at_1pct = None
    best_energy = float('inf')
    best_error = float('inf')

    print(f"\n{'Epoch':>8} {'Energy':>12} {'E_std':>10} {'Accept':>8} "
          f"{'Error%':>8} {'Time':>10}")
    print("-" * 70)

    for epoch in range(max_epochs):
        # Check wall time
        elapsed = time.time() - start_time
        if elapsed > MAX_WALL_TIME:
            print(f"\nReached max wall time ({MAX_WALL_TIME/3600:.1f} hours)")
            break

        # Sample from |psi|^2
        x = sampler.sample(n_steps=mcmc_steps)

        # Reset walkers that drifted too far
        sample_radius = torch.sqrt(torch.sum(x ** 2, dim=-1))
        too_far = sample_radius > max_sample_radius
        if too_far.any():
            n_reset = too_far.sum().item()
            sampler.walkers[too_far] = torch.randn(n_reset, 6, device=device)
            x = sampler.walkers.clone()

        x = x.requires_grad_(True)

        # Compute local energy
        E_loc = compute_local_energy(model, x, harmonic_potential_nd)

        # Forward pass for gradient
        f = model(x)

        # Statistics
        E_mean = E_loc.mean()
        E_std = E_loc.std()

        # REINFORCE loss
        centered_E = (E_loc - E_mean).detach()
        loss = 2.0 * torch.mean(centered_E * f)

        # Update with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Record
        E_val = E_mean.item()
        energies.append(E_val)
        energy_stds.append(E_std.item())
        acceptance_rates.append(sampler.acceptance_rate)

        error_pct = abs(E_val - E_EXACT) / E_EXACT * 100

        if error_pct < best_error:
            best_error = error_pct
            best_energy = E_val

        # Check if reached 1% error
        if epoch_at_1pct is None and error_pct < 1.0:
            epoch_at_1pct = epoch
            print(f"\n*** Reached 1% error at epoch {epoch}! ***\n")

        # Print every 100 epochs
        if epoch % 100 == 0:
            hours = elapsed / 3600
            print(f"{epoch:8d} {E_val:12.6f} {E_std.item():10.4f} "
                  f"{sampler.acceptance_rate:8.2f} {error_pct:8.2f}% "
                  f"{hours:9.2f}h")

        # Adaptive step size
        if epoch > 0 and epoch % 500 == 0:
            avg_accept = np.mean(acceptance_rates[-500:])
            if avg_accept < 0.30:
                sampler.step_size *= 0.9
                print(f"  [Step size decreased to {sampler.step_size:.3f}]")
            elif avg_accept > 0.70:
                sampler.step_size *= 1.1
                print(f"  [Step size increased to {sampler.step_size:.3f}]")

        # Early convergence check
        if epoch > 1000:
            recent_std = np.std(energies[-500:])
            if recent_std < 0.001 and error_pct < 0.5:
                print(f"\nConverged at epoch {epoch} (energy std = {recent_std:.6f})")
                break

    total_time = time.time() - start_time
    final_epochs = len(energies)

    # ========================================
    # Final Results
    # ========================================
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    final_E = energies[-1]
    final_std = energy_stds[-1]
    final_error = abs(final_E - E_EXACT) / E_EXACT * 100

    print(f"\n| Metric                     | Value           |")
    print(f"|----------------------------|-----------------|")
    print(f"| Final energy               | {final_E:15.6f} |")
    print(f"| Exact energy               | {E_EXACT:15.6f} |")
    print(f"| Error                      | {final_error:14.4f}% |")
    print(f"| Energy std (E_loc)         | {final_std:15.6f} |")
    print(f"| Total epochs               | {final_epochs:15d} |")
    print(f"| Training time              | {total_time/60:13.1f} min |")
    print(f"| Epochs to 1% error         | {epoch_at_1pct if epoch_at_1pct else 'N/A':>15} |")
    print(f"| Best energy achieved       | {best_energy:15.6f} |")
    print(f"| Best error achieved        | {best_error:14.4f}% |")

    # ========================================
    # Create convergence plot
    # ========================================
    print("\nGenerating convergence plot...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    epochs_arr = np.arange(len(energies))

    # Energy convergence
    ax1 = axes[0, 0]
    ax1.plot(epochs_arr, energies, 'b-', alpha=0.7, linewidth=0.5)
    ax1.axhline(E_EXACT, color='r', linestyle='--', linewidth=2, label=f'Exact: {E_EXACT}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Error percentage
    ax2 = axes[0, 1]
    errors = np.abs(np.array(energies) - E_EXACT) / E_EXACT * 100
    ax2.semilogy(epochs_arr, errors, 'g-', alpha=0.7, linewidth=0.5)
    ax2.axhline(1.0, color='r', linestyle='--', label='1% target')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error (%)')
    ax2.set_title('Relative Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Energy std
    ax3 = axes[1, 0]
    ax3.semilogy(epochs_arr, energy_stds, 'm-', alpha=0.7, linewidth=0.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('E_loc std (log)')
    ax3.set_title('Local Energy Variance')
    ax3.grid(True, alpha=0.3)

    # Acceptance rate
    ax4 = axes[1, 1]
    ax4.plot(epochs_arr, acceptance_rates, 'c-', alpha=0.7, linewidth=0.5)
    ax4.axhline(0.3, color='r', linestyle='--', alpha=0.5)
    ax4.axhline(0.7, color='r', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Acceptance Rate')
    ax4.set_title('MCMC Acceptance Rate')
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'NNQS 6D Separable Oscillator: E = {final_E:.4f} ({final_error:.2f}% error)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('nnqs_6d_separable_convergence.png', dpi=150, bbox_inches='tight')
    print("Saved: nnqs_6d_separable_convergence.png")

    # Validation
    print("\n" + "-" * 70)
    print("Validation")
    print("-" * 70)
    passed = final_error < 1.0
    print(f"Energy within 1% of exact: {'PASS' if passed else 'FAIL'}")

    return {
        'final_energy': final_E,
        'final_error': final_error,
        'final_std': final_std,
        'epochs': final_epochs,
        'time_seconds': total_time,
        'epoch_at_1pct': epoch_at_1pct,
        'best_energy': best_energy,
        'best_error': best_error,
    }


if __name__ == '__main__':
    result = train_6d_separable()
