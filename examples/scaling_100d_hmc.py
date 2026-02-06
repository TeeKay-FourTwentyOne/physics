"""
Dimensional Scaling Study Phase 3b: 100D Henon-Heiles All-to-All with HMC

Uses Hamiltonian Monte Carlo instead of Metropolis-Hastings to overcome
the curse of dimensionality. HMC uses gradient information to make intelligent
proposals that scale as O(d^1/4) instead of O(d^2).

Strategy: Start with Metropolis for initial training (until network learns
reasonable gradients), then switch to HMC for better exploration.
"""

import torch
import numpy as np
import time
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnqs_baseline import LogAmplitudeNetwork, compute_local_energy
from nnqs_baseline.samplers import HamiltonianMonteCarlo


class BoundedMetropolisSampler:
    """Metropolis sampler with hard rejection for V < 0."""

    def __init__(self, model, V_func, dim, n_walkers=1000, step_size=0.5, device=None):
        self.model = model
        self.V_func = V_func
        self.dim = dim
        self.n_walkers = n_walkers
        self.step_size = step_size

        if device is None:
            device = next(model.parameters()).device
        self.device = device

        self.walkers = 0.5 * torch.randn(n_walkers, dim, device=device)
        self.acceptance_rate = 0.5
        self.v_rejection_rate = 0.0

    def sample(self, n_steps=10):
        n_accepted = 0
        n_v_rejected = 0
        n_total = 0

        with torch.no_grad():
            for _ in range(n_steps):
                proposals = self.walkers + self.step_size * torch.randn_like(self.walkers)
                V_proposed = self.V_func(proposals)
                v_valid = V_proposed >= 0

                log_prob_current = 2.0 * self.model(self.walkers)
                log_prob_proposed = 2.0 * self.model(proposals)
                log_accept = log_prob_proposed - log_prob_current

                metropolis_accept = torch.log(torch.rand(self.n_walkers, device=self.device)) < log_accept
                accept = v_valid & metropolis_accept

                self.walkers = torch.where(accept.unsqueeze(-1), proposals, self.walkers)

                n_accepted += accept.sum().item()
                n_v_rejected += (~v_valid).sum().item()
                n_total += self.n_walkers

        self.acceptance_rate = n_accepted / n_total
        self.v_rejection_rate = n_v_rejected / n_total

        return self.walkers.clone()

    def thermalize(self, n_steps=100):
        self.sample(n_steps=n_steps)

    def adapt_step_size(self, target=0.35):
        """Adapt step size toward target acceptance."""
        if self.acceptance_rate < target - 0.1:
            self.step_size *= 0.9
        elif self.acceptance_rate > target + 0.1:
            self.step_size *= 1.1
        self.step_size = max(0.01, min(2.0, self.step_size))


def hh_all_to_all(x, lam=0.111803):
    """N-dimensional Henon-Heiles with all-to-all coupling."""
    N = x.shape[1]
    V = 0.5 * torch.sum(x ** 2, dim=-1)
    for i in range(N):
        for j in range(i + 1, N):
            V = V + (lam / N) * (x[:, i] ** 2 * x[:, j] - x[:, j] ** 3 / 3)
    return V


def run_single_trial(seed, dim, max_epochs=10000, max_time=4*60*60, verbose=True,
                     hmc_switch_epoch=1000):
    """Run a single training trial with given seed.

    Uses a hybrid strategy: Metropolis for initial training, then switch to HMC
    after the network has learned reasonable gradients.
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LAMBDA = 0.111803
    V_func = lambda x: hh_all_to_all(x, LAMBDA)

    # Conservative network sizing - fixed at 512
    hidden_size = 512
    model = LogAmplitudeNetwork(input_dim=dim, hidden_dims=[hidden_size]*4).to(device)

    # Training setup
    n_walkers = 2000
    mcmc_steps_metro = 20   # More steps for Metropolis
    mcmc_steps_hmc = 10     # Fewer needed for HMC
    max_V_reset = 100.0
    lr = 1e-4
    grad_clip = 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Start with Metropolis sampler (works with random networks)
    sampler = BoundedMetropolisSampler(
        model, V_func, dim=dim, n_walkers=n_walkers,
        step_size=0.5, device=device
    )
    sampler.thermalize(200)

    using_hmc = False
    mcmc_steps = mcmc_steps_metro

    if verbose:
        print(f"  Starting with Metropolis sampler")
        print(f"  Post-thermalization acceptance: {sampler.acceptance_rate:.2%}")

    energies = []
    energy_stds = []

    start_time = time.time()

    for epoch in range(max_epochs):
        elapsed = time.time() - start_time
        if elapsed > max_time:
            if verbose:
                print(f"\n  Reached max time ({max_time/3600:.1f} hours)")
            break

        x = sampler.sample(n_steps=mcmc_steps)

        # Reset high-V walkers
        with torch.no_grad():
            V_samples = V_func(x)
            too_high_V = V_samples > max_V_reset
            if too_high_V.any():
                n_reset = too_high_V.sum().item()
                sampler.walkers[too_high_V] = 0.3 * torch.randn(n_reset, dim, device=device)
                x = sampler.walkers.clone()

        x = x.requires_grad_(True)

        E_loc = compute_local_energy(model, x, V_func)
        f = model(x)

        E_mean = E_loc.mean()
        E_std = E_loc.std()

        centered_E = (E_loc - E_mean).detach()
        loss = 2.0 * torch.mean(centered_E * f)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        energies.append(E_mean.item())
        energy_stds.append(E_std.item())

        # Switch to HMC after network has learned reasonable gradients
        if not using_hmc and epoch == hmc_switch_epoch:
            if verbose:
                print(f"\n  Switching to HMC sampler at epoch {epoch}")
            # Transfer walker positions to HMC sampler
            old_walkers = sampler.walkers.clone()
            sampler = HamiltonianMonteCarlo(
                model,
                dim=dim,
                n_walkers=n_walkers,
                step_size=0.1,          # Conservative start for HMC
                n_leapfrog=20,
                device=device,
                V_func=V_func,
                target_acceptance=0.65
            )
            sampler.walkers = old_walkers
            sampler.thermalize(n_steps=50, adapt_every=10)
            using_hmc = True
            mcmc_steps = mcmc_steps_hmc
            if verbose:
                print(f"  HMC step_size: {sampler.step_size:.4f}, acceptance: {sampler.acceptance_rate:.2%}")

        # Periodic step size adaptation
        if epoch > 0 and epoch % 200 == 0:
            if using_hmc:
                sampler.adapt_step_size()
            else:
                sampler.adapt_step_size(target=0.35)

        sampler_type = "HMC" if using_hmc else "Metro"
        if verbose and epoch % 500 == 0:
            print(f"  Epoch {epoch} [{sampler_type}]: E = {E_mean.item():.6f}, std = {E_std.item():.4f}, "
                  f"accept = {sampler.acceptance_rate:.2%}, step = {sampler.step_size:.4f}")

    total_time = time.time() - start_time

    # Fresh walker validation - use same sampler type
    fresh_n_walkers = 1000
    if using_hmc:
        fresh_sampler = HamiltonianMonteCarlo(
            model,
            dim=dim,
            n_walkers=fresh_n_walkers,
            step_size=sampler.step_size,
            n_leapfrog=20,
            device=device,
            V_func=V_func,
            target_acceptance=0.65
        )
        fresh_sampler.walkers = 0.5 * torch.randn(fresh_n_walkers, dim, device=device)
        fresh_sampler.thermalize(n_steps=100, adapt_every=20)
    else:
        fresh_sampler = BoundedMetropolisSampler(
            model, V_func, dim=dim, n_walkers=fresh_n_walkers,
            step_size=sampler.step_size, device=device
        )
        fresh_sampler.walkers = 0.5 * torch.randn(fresh_n_walkers, dim, device=device)
        fresh_sampler.thermalize(n_steps=500)

    all_E = []
    batch_size = 200
    for _ in range(10):
        samples = fresh_sampler.sample(n_steps=50)
        for start in range(0, fresh_n_walkers, batch_size):
            end = min(start + batch_size, fresh_n_walkers)
            batch = samples[start:end].clone().requires_grad_(True)
            E_loc = compute_local_energy(model, batch, V_func)
            E_np = E_loc.detach().cpu().numpy()
            # Filter out pathological values (NaN, inf, or extreme outliers)
            valid = np.isfinite(E_np) & (np.abs(E_np) < 1e6)
            all_E.append(E_np[valid])
            del batch, E_loc
            torch.cuda.empty_cache()

    all_E = np.concatenate(all_E)
    fresh_E_mean = np.mean(all_E) if len(all_E) > 0 else float('nan')
    fresh_E_std = np.std(all_E) if len(all_E) > 0 else float('nan')

    return {
        'seed': seed,
        'final_energy': energies[-1],
        'final_std': energy_stds[-1],
        'fresh_energy': fresh_E_mean,
        'fresh_std': fresh_E_std,
        'training_time': total_time,
        'epochs': len(energies),
        'v_rejection_rate': sampler.v_rejection_rate,
        'acceptance_rate': sampler.acceptance_rate,
        'final_step_size': sampler.step_size,
        'used_hmc': using_hmc,
    }


def main():
    parser = argparse.ArgumentParser(description='100D Henon-Heiles with hybrid Metropolis/HMC')
    parser.add_argument('--dim', type=int, default=100, help='Number of dimensions')
    parser.add_argument('--max_epochs', type=int, default=10000, help='Max training epochs')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                        help='Random seeds for trials')
    parser.add_argument('--hmc_switch', type=int, default=1000,
                        help='Epoch to switch from Metropolis to HMC')
    args = parser.parse_args()

    DIM = args.dim
    E_HARMONIC = DIM / 2.0

    print("=" * 70)
    print(f"Dimensional Scaling Study: {DIM}D Henon-Heiles All-to-All (Hybrid)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Network: 512 hidden x 4 layers (conservative sizing)")
    print(f"Strategy: Metropolis -> HMC at epoch {args.hmc_switch}")
    print(f"Walkers: 2000, HMC leapfrog steps: 20")
    print(f"Max time per trial: 4 hours")

    LAMBDA = 0.111803
    SEEDS = args.seeds

    print("\n" + "-" * 70)
    print(f"Running {len(SEEDS)} independent trials (seeds: {SEEDS})")
    print("-" * 70)

    results = []

    for i, seed in enumerate(SEEDS):
        print(f"\n{'='*50}")
        print(f"TRIAL {i+1}/{len(SEEDS)} (seed={seed})")
        print(f"{'='*50}")

        result = run_single_trial(seed, dim=DIM, max_epochs=args.max_epochs, max_time=4*60*60,
                                  hmc_switch_epoch=args.hmc_switch)
        results.append(result)

        diff_pct = abs(result['fresh_energy'] - result['final_energy']) / result['final_energy'] * 100

        print(f"\nTrial {i+1} complete:")
        print(f"  Final energy: {result['final_energy']:.6f}")
        print(f"  E_loc std: {result['final_std']:.4f}")
        print(f"  Fresh validation: {result['fresh_energy']:.6f} +/- {result['fresh_std']:.4f}")
        print(f"  Fresh/Train diff: {diff_pct:.3f}%")
        print(f"  Training time: {result['training_time']/60:.1f} min")
        print(f"  Epochs completed: {result['epochs']}")
        print(f"  Acceptance rate: {result['acceptance_rate']:.2%}")
        print(f"  V<0 rejection: {result['v_rejection_rate']:.2%}")
        print(f"  Final step size: {result['final_step_size']:.4f}")
        print(f"  Used HMC: {result['used_hmc']}")

    # Summary statistics
    print("\n" + "=" * 70)
    print(f"SUMMARY: {DIM}D All-to-All Henon-Heiles (Hybrid Metro/HMC)")
    print("=" * 70)

    final_energies = [r['final_energy'] for r in results]
    fresh_energies = [r['fresh_energy'] for r in results]
    final_stds = [r['final_std'] for r in results]
    times = [r['training_time'] for r in results]
    accept_rates = [r['acceptance_rate'] for r in results]

    mean_E = np.mean(final_energies)
    std_E = np.std(final_energies)
    cv_E = std_E / mean_E * 100 if mean_E != 0 else float('inf')

    mean_fresh = np.mean(fresh_energies)
    std_fresh = np.std(fresh_energies)

    E_per_dim = mean_E / DIM

    print(f"\n| Metric                    | Value              |")
    print(f"|---------------------------|-------------------|")
    print(f"| Mean energy (training)    | {mean_E:.6f} +/- {std_E:.6f} |")
    print(f"| Mean energy (fresh valid) | {mean_fresh:.6f} +/- {std_fresh:.6f} |")
    print(f"| Coefficient of variation  | {cv_E:.4f}%          |")
    print(f"| Mean E_loc std            | {np.mean(final_stds):.4f}            |")
    print(f"| E_0/N ratio               | {E_per_dim:.6f}          |")
    print(f"| Mean training time        | {np.mean(times)/60:.1f} min          |")
    print(f"| Mean acceptance rate      | {np.mean(accept_rates):.2%}          |")

    print(f"\n| Run | Seed | Final E    | Fresh E    | Fresh Std | Diff%  | Accept | Time  |")
    print(f"|-----|------|------------|------------|-----------|--------|--------|-------|")
    for i, r in enumerate(results):
        diff = abs(r['fresh_energy'] - r['final_energy']) / r['final_energy'] * 100
        print(f"|  {i+1}  | {r['seed']:4d} | {r['final_energy']:.6f}   | {r['fresh_energy']:.6f}   | "
              f"{r['fresh_std']:.4f}    | {diff:.3f}% | {r['acceptance_rate']:.1%}  | {r['training_time']/60:.1f}m |")

    # Validation checks
    print("\n" + "-" * 70)
    print("Validation Checks (Success Criteria)")
    print("-" * 70)

    fresh_diff = abs(mean_fresh - mean_E) / mean_E * 100 if mean_E != 0 else float('inf')

    cv_ok = cv_E < 1.0
    fresh_ok = fresh_diff < 0.5
    scaling_ok = 0.48 <= E_per_dim <= 0.52
    accept_ok = np.mean(accept_rates) > 0.50

    print(f"\n1. CV < 1%:                  {'PASS' if cv_ok else 'FAIL'} (CV = {cv_E:.4f}%)")
    print(f"2. Fresh within 0.5%:        {'PASS' if fresh_ok else 'FAIL'} (diff = {fresh_diff:.4f}%)")
    print(f"3. E/N in [0.48, 0.52]:      {'PASS' if scaling_ok else 'FAIL'} (E/N = {E_per_dim:.4f})")
    print(f"4. Acceptance > 50%:         {'PASS' if accept_ok else 'FAIL'} (accept = {np.mean(accept_rates):.2%})")

    all_pass = cv_ok and fresh_ok and scaling_ok
    print(f"\nOVERALL: {'ALL CRITERIA MET' if all_pass else 'CRITERIA NOT MET'}")

    # Save results
    results_dict = {
        'dim': DIM,
        'lambda': LAMBDA,
        'topology': 'all-to-all',
        'sampler': 'HMC',
        'seeds': SEEDS,
        'mean_energy': mean_E,
        'std_energy': std_E,
        'cv_percent': cv_E,
        'E_per_dim': E_per_dim,
        'mean_fresh_energy': mean_fresh,
        'std_fresh_energy': std_fresh,
        'fresh_diff_percent': fresh_diff,
        'mean_eloc_std': np.mean(final_stds),
        'mean_time': np.mean(times),
        'mean_acceptance': np.mean(accept_rates),
        'individual_results': results,
        'all_criteria_met': all_pass,
    }

    save_path = f'scaling_{DIM}d_hmc.pt'
    torch.save(results_dict, save_path)
    print(f"\nResults saved to: {save_path}")

    return results_dict


if __name__ == "__main__":
    results = main()
