"""
NNQS Phase 4 Fix: 6D Henon-Heiles with Hard V < 0 Rejection

Quick fix: Reject any MCMC move that would put V(x) < 0.
This keeps walkers confined to the potential well.
"""

import torch
import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnqs_baseline import LogAmplitudeNetwork, compute_local_energy


def henon_heiles_potential(x, lam=0.111803):
    """6D Henon-Heiles potential with periodic boundary."""
    V = 0.5 * torch.sum(x ** 2, dim=-1)
    for i in range(6):
        j = (i + 1) % 6
        V = V + lam * (x[:, i] ** 2 * x[:, j] - x[:, j] ** 3 / 3)
    return V


class BoundedMetropolisSampler:
    """
    Metropolis sampler with hard rejection for V < 0.

    Any proposed move where V(x_new) < 0 is automatically rejected.
    """

    def __init__(self, model, V_func, dim, n_walkers=1000, step_size=0.5, device=None):
        self.model = model
        self.V_func = V_func
        self.dim = dim
        self.n_walkers = n_walkers
        self.step_size = step_size

        if device is None:
            device = next(model.parameters()).device
        self.device = device

        # Initialize walkers in the well (small positions)
        self.walkers = 0.5 * torch.randn(n_walkers, dim, device=device)

        # Statistics
        self.acceptance_rate = 0.5
        self.v_rejection_rate = 0.0  # Fraction rejected due to V < 0

    def sample(self, n_steps=10):
        """Run MCMC with hard V < 0 rejection."""
        n_accepted = 0
        n_v_rejected = 0
        n_total = 0

        with torch.no_grad():
            for _ in range(n_steps):
                # Propose moves
                proposals = self.walkers + self.step_size * torch.randn_like(self.walkers)

                # Compute V at proposals
                V_proposed = self.V_func(proposals)

                # Hard rejection: V < 0 means automatic reject
                v_valid = V_proposed >= 0

                # For valid proposals, do normal Metropolis
                log_prob_current = 2.0 * self.model(self.walkers)
                log_prob_proposed = 2.0 * self.model(proposals)
                log_accept = log_prob_proposed - log_prob_current

                # Metropolis acceptance (only for V >= 0 proposals)
                metropolis_accept = torch.log(torch.rand(self.n_walkers, device=self.device)) < log_accept

                # Final acceptance: V >= 0 AND Metropolis accepts
                accept = v_valid & metropolis_accept

                # Update walkers
                self.walkers = torch.where(accept.unsqueeze(-1), proposals, self.walkers)

                # Statistics
                n_accepted += accept.sum().item()
                n_v_rejected += (~v_valid).sum().item()
                n_total += self.n_walkers

        self.acceptance_rate = n_accepted / n_total
        self.v_rejection_rate = n_v_rejected / n_total

        return self.walkers.clone()

    def thermalize(self, n_steps=100):
        """Burn-in."""
        self.sample(n_steps=n_steps)


def main():
    print("=" * 70)
    print("NNQS Phase 4 Fix: 6D Henon-Heiles with Hard V < 0 Rejection")
    print("=" * 70)

    E_EXPECTED = 2.99
    LAMBDA = 0.111803
    MAX_EPOCHS = 10000
    MAX_TIME = 15 * 60  # 15 minutes for quick diagnostic

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize network
    print("\n" + "-" * 70)
    print("Initializing network (random weights)")
    print("-" * 70)

    model = LogAmplitudeNetwork(input_dim=6, hidden_dims=[512, 512, 512, 512]).to(device)
    V_func = lambda x: henon_heiles_potential(x, LAMBDA)

    # Check initial state
    test_x = torch.randn(1000, 6, device=device) * 0.5  # Small positions
    test_x = test_x.requires_grad_(True)
    E_loc_init = compute_local_energy(model, test_x, V_func)
    print(f"Initial E_loc mean: {E_loc_init.mean().item():.4f}")
    print(f"Initial E_loc std: {E_loc_init.std().item():.4f}")

    # Training setup
    print("\n" + "-" * 70)
    print("VMC Training with Bounded Sampler")
    print("-" * 70)

    n_walkers = 10000
    mcmc_steps = 20
    max_V_reset = 20.0  # Tighter reset threshold
    lr = 1e-4
    grad_clip = 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sampler = BoundedMetropolisSampler(model, V_func, dim=6, n_walkers=n_walkers,
                                        step_size=0.5, device=device)

    print("Thermalizing sampler...")
    sampler.thermalize(200)
    print(f"Initial acceptance: {sampler.acceptance_rate:.2%}, V<0 rejections: {sampler.v_rejection_rate:.2%}")

    # Training history
    energies = []
    energy_stds = []
    acceptance_rates = []
    v_rejection_rates = []

    best_energy = float('inf')
    best_error = float('inf')
    best_epoch = 0
    epoch_at_1pct = None

    start_time = time.time()

    print(f"\n{'Epoch':>6} {'Energy':>10} {'E_std':>8} {'Accept':>8} {'V<0 Rej':>8} {'Error%':>8} {'Time':>8}")
    print("-" * 70)

    for epoch in range(MAX_EPOCHS):
        elapsed = time.time() - start_time
        if elapsed > MAX_TIME:
            print(f"\nReached max time ({MAX_TIME/60:.0f} minutes)")
            break

        # Sample with bounded sampler
        x = sampler.sample(n_steps=mcmc_steps)

        # Additional reset for high V (tighter threshold)
        with torch.no_grad():
            V_samples = V_func(x)
            too_high_V = V_samples > max_V_reset
            if too_high_V.any():
                n_reset = too_high_V.sum().item()
                sampler.walkers[too_high_V] = 0.3 * torch.randn(n_reset, 6, device=device)
                x = sampler.walkers.clone()

        x = x.requires_grad_(True)

        # Compute local energy
        E_loc = compute_local_energy(model, x, V_func)
        f = model(x)

        E_mean = E_loc.mean()
        E_std = E_loc.std()

        # REINFORCE loss
        centered_E = (E_loc - E_mean).detach()
        loss = 2.0 * torch.mean(centered_E * f)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # Record
        E_val = E_mean.item()
        energies.append(E_val)
        energy_stds.append(E_std.item())
        acceptance_rates.append(sampler.acceptance_rate)
        v_rejection_rates.append(sampler.v_rejection_rate)

        error_pct = abs(E_val - E_EXPECTED) / E_EXPECTED * 100

        if error_pct < best_error:
            best_error = error_pct
            best_energy = E_val
            best_epoch = epoch

        if epoch_at_1pct is None and error_pct < 1.0:
            epoch_at_1pct = epoch
            mins = elapsed / 60
            print(f"\n*** Reached 1% error at epoch {epoch} ({mins:.1f} min)! ***\n")

        # Print every 100 epochs
        if epoch % 100 == 0:
            mins = elapsed / 60
            print(f"{epoch:6d} {E_val:10.4f} {E_std.item():8.4f} "
                  f"{sampler.acceptance_rate:7.2%} {sampler.v_rejection_rate:7.2%} "
                  f"{error_pct:8.2f}% {mins:7.1f}m")

        # Adaptive step size
        if epoch > 0 and epoch % 200 == 0:
            avg_accept = np.mean(acceptance_rates[-200:])
            if avg_accept < 0.15:
                sampler.step_size *= 0.85
                print(f"  [Step size -> {sampler.step_size:.3f}]")
            elif avg_accept > 0.50:
                sampler.step_size *= 1.15
                print(f"  [Step size -> {sampler.step_size:.3f}]")

    total_time = time.time() - start_time
    final_epochs = len(energies)

    # Results
    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)

    final_E = energies[-1]
    final_std = energy_stds[-1]
    final_error = abs(final_E - E_EXPECTED) / E_EXPECTED * 100
    avg_accept = np.mean(acceptance_rates[-500:]) if len(acceptance_rates) >= 500 else np.mean(acceptance_rates)
    avg_v_reject = np.mean(v_rejection_rates[-500:]) if len(v_rejection_rates) >= 500 else np.mean(v_rejection_rates)

    print(f"\n| Metric                     | Value           |")
    print(f"|----------------------------|-----------------|")
    print(f"| Final energy               | {final_E:15.6f} |")
    print(f"| Best energy                | {best_energy:15.6f} |")
    print(f"| Expected energy            | {E_EXPECTED:15.6f} |")
    print(f"| Final error                | {final_error:14.2f}% |")
    print(f"| Best error                 | {best_error:14.2f}% |")
    print(f"| Final E_loc std            | {final_std:15.4f} |")
    print(f"| Epochs to 1% error         | {str(epoch_at_1pct) if epoch_at_1pct else 'N/A':>15} |")
    print(f"| Training time              | {total_time/60:13.1f} min |")
    print(f"| Avg acceptance rate        | {avg_accept:14.2%} |")
    print(f"| Avg V<0 rejection rate     | {avg_v_reject:14.2%} |")

    # Fresh Walker Validation
    print("\n" + "-" * 70)
    print("Fresh Walker Validation (with bounded sampler)")
    print("-" * 70)

    fresh_sampler = BoundedMetropolisSampler(model, V_func, dim=6, n_walkers=5000,
                                              step_size=0.4, device=device)
    fresh_sampler.walkers = 0.5 * torch.randn(5000, 6, device=device)

    # Burn-in
    print("Running burn-in...")
    fresh_sampler.sample(n_steps=500)
    print(f"  Acceptance: {fresh_sampler.acceptance_rate:.2%}, V<0 rejections: {fresh_sampler.v_rejection_rate:.2%}")

    # Sample
    print("Sampling...")
    all_E = []
    all_V = []
    for i in range(15):
        samples = fresh_sampler.sample(n_steps=100)
        samples = samples.requires_grad_(True)
        E_loc = compute_local_energy(model, samples, V_func)
        V = V_func(samples.detach())
        all_E.append(E_loc.detach().cpu().numpy())
        all_V.append(V.cpu().numpy())

    all_E = np.concatenate(all_E)
    all_V = np.concatenate(all_V)

    fresh_E_mean = np.mean(all_E)
    fresh_E_std = np.std(all_E)
    fresh_V_mean = np.mean(all_V)
    fresh_V_min = np.min(all_V)
    fresh_V_max = np.max(all_V)

    print(f"\n| Metric                  | Measured   | Expected   |")
    print(f"|-------------------------|------------|------------|")
    print(f"| E_loc mean              | {fresh_E_mean:10.4f} | {E_EXPECTED:10.4f} |")
    print(f"| E_loc std               | {fresh_E_std:10.4f} |    <0.5    |")
    print(f"| V mean                  | {fresh_V_mean:10.4f} |    ~1-2    |")
    print(f"| V min                   | {fresh_V_min:10.4f} |    >=0     |")
    print(f"| V max                   | {fresh_V_max:10.4f} |    <20     |")

    fresh_error = abs(fresh_E_mean - E_EXPECTED) / E_EXPECTED * 100
    print(f"\nFresh walker energy error: {fresh_error:.2f}%")

    # Diagnostic summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    stable = final_std < 1.0 and fresh_E_std < 1.0
    converged = best_error < 1.0
    validated = fresh_error < 5.0

    print(f"\n1. Training stabilized?     {'YES' if stable else 'NO'} (E_loc std = {final_std:.4f})")
    print(f"2. Reached 1% error?        {'YES' if converged else 'NO'} (best = {best_error:.2f}%)")
    print(f"3. Fresh walker validates?  {'YES' if validated else 'NO'} (error = {fresh_error:.2f}%)")
    print(f"4. V<0 rejection helped?    Rejected {avg_v_reject:.1%} of proposals")

    if stable and converged and validated:
        print("\nSUCCESS: Hard V<0 rejection fixed the instability!")
    elif stable:
        print("\nPARTIAL: Training stabilized but accuracy not achieved.")
        print("May need more epochs or different hyperparameters.")
    else:
        print("\nFAILED: Need a different approach (importance sampling, different architecture, etc.)")

    # Quick convergence plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs_arr = np.arange(len(energies))

    ax1 = axes[0, 0]
    ax1.plot(epochs_arr, energies, 'b-', alpha=0.7, linewidth=0.5)
    ax1.axhline(E_EXPECTED, color='r', linestyle='--', linewidth=2, label=f'Expected: {E_EXPECTED}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.semilogy(epochs_arr, energy_stds, 'm-', alpha=0.7, linewidth=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('E_loc std')
    ax2.set_title('Local Energy Variance')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(epochs_arr, acceptance_rates, 'c-', alpha=0.7, linewidth=0.5, label='Acceptance')
    ax3.plot(epochs_arr, v_rejection_rates, 'r-', alpha=0.7, linewidth=0.5, label='V<0 Rejection')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Rate')
    ax3.set_title('MCMC Statistics')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.hist(all_V, bins=50, density=True, alpha=0.7, color='purple')
    ax4.axvline(0, color='r', linestyle='--', label='V=0 boundary')
    ax4.set_xlabel('V(x)')
    ax4.set_ylabel('Density')
    ax4.set_title(f'Potential Distribution (min={fresh_V_min:.2f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'NNQS 6D HH (Bounded): E = {final_E:.4f} ({final_error:.2f}% error)', fontsize=12)
    plt.tight_layout()
    plt.savefig('nnqs_6d_hh_bounded_convergence.png', dpi=150, bbox_inches='tight')
    print("\nSaved: nnqs_6d_hh_bounded_convergence.png")

    return stable and converged and validated


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
