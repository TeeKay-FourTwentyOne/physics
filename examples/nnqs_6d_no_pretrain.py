"""
NNQS 6D: Training WITHOUT Gaussian Pretraining

Test whether VMC can actually learn from random initialization.
This is critical for Phase 3 where the ground state is NOT Gaussian.
"""

import torch
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnqs_baseline import LogAmplitudeNetwork, MetropolisSampler, compute_local_energy
from nnqs_baseline.train import harmonic_potential_nd


def main():
    print("=" * 70)
    print("NNQS 6D: Training from RANDOM Initialization (No Pretraining)")
    print("=" * 70)

    E_EXACT = 3.0
    MAX_EPOCHS = 5000
    MAX_TIME = 30 * 60  # 30 minutes

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    # ============================================================
    # Initialize network with RANDOM weights (no pretraining!)
    # ============================================================
    print("\n" + "-" * 70)
    print("Initializing network with RANDOM weights (NO pretraining)")
    print("-" * 70)

    model = LogAmplitudeNetwork(input_dim=6, hidden_dims=[512, 512, 512, 512]).to(device)

    # Verify the network starts with essentially random output
    test_x = torch.randn(1000, 6, device=device)
    with torch.no_grad():
        f_init = model(test_x)
    print(f"Initial f(x) mean: {f_init.mean().item():.4f}")
    print(f"Initial f(x) std: {f_init.std().item():.4f}")
    print("(Should be near 0 due to zero-initialized final layer)")

    # Check initial energy estimate
    test_x = test_x.requires_grad_(True)
    E_loc_init = compute_local_energy(model, test_x, harmonic_potential_nd)
    print(f"\nInitial E_loc mean: {E_loc_init.mean().item():.4f}")
    print(f"Initial E_loc std: {E_loc_init.std().item():.4f}")

    # ============================================================
    # VMC Training (same settings as Phase 2)
    # ============================================================
    print("\n" + "-" * 70)
    print("VMC Training (no pretraining)")
    print("-" * 70)

    n_walkers = 10000
    mcmc_steps = 20
    max_sample_radius = 15.0
    lr = 5e-4
    grad_clip = 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sampler = MetropolisSampler(model, dim=6, n_walkers=n_walkers, step_size=0.5, device=device)

    # Thermalize sampler
    print("Thermalizing sampler...")
    sampler.thermalize(200)

    # Training history
    energies = []
    energy_stds = []
    acceptance_rates = []

    epoch_at_1pct = None
    epoch_at_5pct = None
    start_time = time.time()

    print(f"\n{'Epoch':>6} {'Energy':>10} {'E_std':>8} {'Accept':>8} {'Error%':>8} {'Time':>8}")
    print("-" * 60)

    for epoch in range(MAX_EPOCHS):
        elapsed = time.time() - start_time
        if elapsed > MAX_TIME:
            print(f"\nReached max time ({MAX_TIME/60:.0f} minutes)")
            break

        # Sample
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

        error_pct = abs(E_val - E_EXACT) / E_EXACT * 100

        # Track milestones
        if epoch_at_5pct is None and error_pct < 5.0:
            epoch_at_5pct = epoch
            print(f"\n*** Reached 5% error at epoch {epoch}! ***\n")

        if epoch_at_1pct is None and error_pct < 1.0:
            epoch_at_1pct = epoch
            print(f"\n*** Reached 1% error at epoch {epoch}! ***\n")

        # Print every 100 epochs
        if epoch % 100 == 0:
            mins = elapsed / 60
            print(f"{epoch:6d} {E_val:10.4f} {E_std.item():8.4f} "
                  f"{sampler.acceptance_rate:7.2%} {error_pct:8.2f}% {mins:7.1f}m")

        # Adaptive step size
        if epoch > 0 and epoch % 500 == 0:
            avg_accept = np.mean(acceptance_rates[-500:])
            if avg_accept < 0.20:
                sampler.step_size *= 0.8
                print(f"  [Step size decreased to {sampler.step_size:.3f}]")
            elif avg_accept > 0.80:
                sampler.step_size *= 1.2
                print(f"  [Step size increased to {sampler.step_size:.3f}]")

    total_time = time.time() - start_time
    final_epochs = len(energies)

    # ============================================================
    # Final Results
    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS: Training from Random Initialization")
    print("=" * 70)

    final_E = energies[-1]
    final_std = energy_stds[-1]
    final_error = abs(final_E - E_EXACT) / E_EXACT * 100
    avg_accept_last_500 = np.mean(acceptance_rates[-500:]) if len(acceptance_rates) >= 500 else np.mean(acceptance_rates)

    print(f"\n| Metric                     | Value           |")
    print(f"|----------------------------|-----------------|")
    print(f"| Final energy               | {final_E:15.6f} |")
    print(f"| Exact energy               | {E_EXACT:15.6f} |")
    print(f"| Final error                | {final_error:14.2f}% |")
    print(f"| Final E_loc std            | {final_std:15.4f} |")
    print(f"| Total epochs               | {final_epochs:15d} |")
    print(f"| Training time              | {total_time/60:13.1f} min |")
    print(f"| Epochs to 5% error         | {epoch_at_5pct if epoch_at_5pct else 'N/A':>15} |")
    print(f"| Epochs to 1% error         | {epoch_at_1pct if epoch_at_1pct else 'N/A':>15} |")
    print(f"| Avg acceptance (last 500)  | {avg_accept_last_500:14.2%} |")

    # ============================================================
    # Validation with fresh walkers
    # ============================================================
    print("\n" + "-" * 70)
    print("Validation: Fresh Walker Test")
    print("-" * 70)

    fresh_walkers = torch.randn(5000, 6, device=device)

    # Find good step size
    def test_acceptance(walkers, step_size, n_steps=100):
        walkers = walkers.clone()
        n_acc = 0
        with torch.no_grad():
            for _ in range(n_steps):
                proposals = walkers + step_size * torch.randn_like(walkers)
                log_p_curr = 2.0 * model(walkers)
                log_p_prop = 2.0 * model(proposals)
                accept = torch.log(torch.rand(len(walkers), device=device)) < (log_p_prop - log_p_curr)
                walkers = torch.where(accept.unsqueeze(-1), proposals, walkers)
                n_acc += accept.sum().item()
        return n_acc / (n_steps * len(walkers)), walkers

    # Tune step size
    best_ss = 0.5
    for ss in [0.3, 0.5, 0.7, 1.0]:
        acc, _ = test_acceptance(fresh_walkers, ss)
        if 0.3 < acc < 0.7:
            best_ss = ss
            break

    # Run MCMC
    _, walkers = test_acceptance(fresh_walkers, best_ss, n_steps=500)  # burn-in

    # Sample
    all_E = []
    for _ in range(10):
        _, walkers = test_acceptance(walkers, best_ss, n_steps=100)
        w = walkers.clone().requires_grad_(True)
        E_loc = compute_local_energy(model, w, harmonic_potential_nd)
        all_E.append(E_loc.detach().cpu().numpy())

    all_E = np.concatenate(all_E)
    fresh_E_mean = np.mean(all_E)
    fresh_E_std = np.std(all_E)

    print(f"\nFresh walker test (step_size={best_ss}):")
    print(f"  E_loc mean: {fresh_E_mean:.4f} (expected: 3.0)")
    print(f"  E_loc std:  {fresh_E_std:.4f} (expected: <0.1)")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    converged = final_error < 1.0
    fresh_valid = abs(fresh_E_mean - 3.0) < 0.2

    if converged and fresh_valid:
        print("\nSUCCESS: VMC can learn from random initialization!")
        print("Ready for Phase 3 (coupled oscillator).")
    elif converged:
        print("\nPARTIAL: Training converged but fresh walker test shows issues.")
        print("May need longer training or different hyperparameters.")
    else:
        print(f"\nFAILED: Training did not converge to 1% error.")
        print(f"Final error: {final_error:.2f}%")
        print("Need to fix training loop before Phase 3.")

    return converged and fresh_valid


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
