"""
NNQS Phase 1: 1D Harmonic Oscillator Validation

This script validates the Neural Network Quantum State implementation
on the 1D harmonic oscillator where the exact solution is known:
    H = -1/2 * d^2/dx^2 + 1/2 * x^2
    E_exact = 0.5
    psi_exact = exp(-x^2/2) / pi^(1/4)

Validation criteria:
1. Energy converges to 0.5 within 1%
2. Variance of E_loc decreases during training
3. Samples cluster around origin (Gaussian distribution)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from nnqs_baseline import NNQSTrainer
from nnqs_baseline.train import harmonic_potential_1d


def exact_psi_squared(x):
    """Exact |psi|^2 for ground state of 1D harmonic oscillator."""
    return np.exp(-x**2) / np.sqrt(np.pi)


def main():
    print("=" * 70)
    print("NNQS Phase 1: 1D Harmonic Oscillator Validation")
    print("=" * 70)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create trainer
    trainer = NNQSTrainer(
        V_func=harmonic_potential_1d,
        dim=1,
        hidden_dims=[64, 64],
        device=device
    )

    # Count parameters
    n_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"Network parameters: {n_params:,}")

    # Train
    print("\n" + "-" * 70)
    print("Training NNQS...")
    print("-" * 70)

    result = trainer.train(
        n_epochs=5000,
        n_walkers=1000,
        lr=1e-3,
        pretrain=True,
        pretrain_epochs=500,
        verbose=True,
        log_interval=500
    )

    # Final evaluation with more samples
    print("\n" + "-" * 70)
    print("Final Evaluation")
    print("-" * 70)

    E_final, E_err = trainer.evaluate_energy(n_samples=10000)
    E_exact = 0.5

    print(f"Final energy: {E_final:.6f} +/- {E_err:.6f}")
    print(f"Exact energy: {E_exact:.6f}")
    print(f"Relative error: {abs(E_final - E_exact) / E_exact * 100:.4f}%")

    # Validation checks
    print("\n" + "-" * 70)
    print("Validation Checks")
    print("-" * 70)

    # 1. Energy accuracy
    energy_error = abs(E_final - E_exact) / E_exact
    energy_ok = energy_error < 0.01
    print(f"1. Energy within 1%: {'PASS' if energy_ok else 'FAIL'} "
          f"(error = {energy_error*100:.2f}%)")

    # 2. Variance decreased
    early_std = np.mean(result['energy_stds'][:100])
    late_std = np.mean(result['energy_stds'][-100:])
    variance_ok = late_std < early_std
    print(f"2. E_loc variance decreased: {'PASS' if variance_ok else 'FAIL'} "
          f"({early_std:.4f} -> {late_std:.4f})")

    # 3. Sample distribution
    samples = trainer.get_samples(5000)
    sample_mean = samples.mean()
    sample_std = samples.std()
    # For |psi|^2 = exp(-x^2)/sqrt(pi), mean=0, var=0.5
    mean_ok = abs(sample_mean) < 0.1
    var_ok = abs(sample_std**2 - 0.5) < 0.1
    print(f"3. Sample mean ~0: {'PASS' if mean_ok else 'FAIL'} "
          f"(mean = {sample_mean:.4f})")
    print(f"   Sample var ~0.5: {'PASS' if var_ok else 'FAIL'} "
          f"(var = {sample_std**2:.4f})")

    all_passed = energy_ok and variance_ok and mean_ok and var_ok
    print("\n" + "=" * 70)
    print(f"OVERALL: {'ALL VALIDATION CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print("=" * 70)

    # Create plots
    print("\nGenerating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Energy convergence
    ax1 = axes[0, 0]
    epochs = np.arange(len(result['energies']))
    ax1.plot(epochs, result['energies'], 'b-', alpha=0.7, linewidth=0.5)
    ax1.axhline(E_exact, color='r', linestyle='--', linewidth=2, label=f'Exact: {E_exact}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Zoom inset
    ax1_inset = ax1.inset_axes([0.5, 0.5, 0.45, 0.4])
    ax1_inset.plot(epochs[-1000:], result['energies'][-1000:], 'b-', alpha=0.7, linewidth=0.5)
    ax1_inset.axhline(E_exact, color='r', linestyle='--', linewidth=1)
    ax1_inset.set_title('Last 1000 epochs', fontsize=9)
    ax1_inset.grid(True, alpha=0.3)

    # Plot 2: Energy standard deviation
    ax2 = axes[0, 1]
    ax2.semilogy(epochs, result['energy_stds'], 'g-', alpha=0.7, linewidth=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Energy Std Dev (log scale)')
    ax2.set_title('Local Energy Variance')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Sample histogram
    ax3 = axes[1, 0]
    ax3.hist(samples.flatten(), bins=100, density=True, alpha=0.7, label='MCMC samples')
    x_range = np.linspace(-4, 4, 200)
    ax3.plot(x_range, exact_psi_squared(x_range), 'r-', linewidth=2, label='Exact |psi|^2')
    ax3.set_xlabel('x')
    ax3.set_ylabel('Probability density')
    ax3.set_title('Sample Distribution vs Exact')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Learned wavefunction
    ax4 = axes[1, 1]
    x_plot = torch.linspace(-4, 4, 200, device=device).unsqueeze(-1)
    with torch.no_grad():
        f = trainer.model(x_plot)
        psi_sq_learned = torch.exp(2 * f).cpu().numpy()
        # Normalize
        dx = 8 / 200
        psi_sq_learned = psi_sq_learned / (psi_sq_learned.sum() * dx)

    x_np = x_plot.cpu().numpy().flatten()
    ax4.plot(x_np, psi_sq_learned, 'b-', linewidth=2, label='NNQS |psi|^2')
    ax4.plot(x_np, exact_psi_squared(x_np), 'r--', linewidth=2, label='Exact |psi|^2')
    ax4.set_xlabel('x')
    ax4.set_ylabel('|psi(x)|^2')
    ax4.set_title('Learned vs Exact Wavefunction')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nnqs_1d_validation.png', dpi=150, bbox_inches='tight')
    print("Saved: nnqs_1d_validation.png")

    plt.show()

    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
