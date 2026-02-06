"""
Dimensional Scaling Study Phase 2b: 24D Henon-Heiles All-to-All

Uses all-to-all coupling with 1/N normalization.
"""

import torch
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnqs_baseline import LogAmplitudeNetwork, compute_local_energy


def hh_all_to_all(x, lam=0.111803):
    """
    N-dimensional Henon-Heiles with all-to-all coupling.

    V = 0.5 * sum(x_i^2) + (lam/N) * sum_{i<j} (x_i^2 * x_j - x_j^3/3)

    The 1/N normalization keeps anharmonic contribution O(1) as N grows.
    """
    N = x.shape[1]

    # Harmonic base
    V = 0.5 * torch.sum(x ** 2, dim=-1)

    # All-to-all anharmonic coupling (normalized by N)
    for i in range(N):
        for j in range(i + 1, N):
            V = V + (lam / N) * (x[:, i] ** 2 * x[:, j] - x[:, j] ** 3 / 3)

    return V


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


def run_single_trial(seed, dim, max_epochs=10000, max_time=45*60, verbose=True):
    """Run a single training trial with given seed."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    LAMBDA = 0.111803
    V_func = lambda x: hh_all_to_all(x, LAMBDA)

    # Network - conservative size for memory
    hidden_size = 512  # Keep same as 6D to fit in GPU memory
    model = LogAmplitudeNetwork(input_dim=dim, hidden_dims=[hidden_size, hidden_size, hidden_size, hidden_size]).to(device)

    # Training setup - reduce walkers for memory
    n_walkers = 5000  # Reduced from 10000
    mcmc_steps = 20
    max_V_reset = 20.0 + 2.0 * (dim - 6)  # Scale reset threshold
    lr = 1e-4
    grad_clip = 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sampler = BoundedMetropolisSampler(model, V_func, dim=dim, n_walkers=n_walkers,
                                        step_size=0.5, device=device)
    sampler.thermalize(200)

    energies = []
    energy_stds = []

    start_time = time.time()

    for epoch in range(max_epochs):
        elapsed = time.time() - start_time
        if elapsed > max_time:
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

        # Adaptive step size
        if epoch > 0 and epoch % 200 == 0:
            if sampler.acceptance_rate < 0.15:
                sampler.step_size *= 0.85
            elif sampler.acceptance_rate > 0.50:
                sampler.step_size *= 1.15

        if verbose and epoch % 500 == 0:
            print(f"  Epoch {epoch}: E = {E_mean.item():.6f}, std = {E_std.item():.4f}, "
                  f"accept = {sampler.acceptance_rate:.2%}")

    total_time = time.time() - start_time

    # Fresh walker validation
    fresh_sampler = BoundedMetropolisSampler(model, V_func, dim=dim, n_walkers=5000,
                                              step_size=0.4, device=device)
    fresh_sampler.walkers = 0.5 * torch.randn(5000, dim, device=device)
    fresh_sampler.sample(n_steps=500)  # burn-in

    all_E = []
    for _ in range(10):
        samples = fresh_sampler.sample(n_steps=100)
        samples = samples.requires_grad_(True)
        E_loc = compute_local_energy(model, samples, V_func)
        all_E.append(E_loc.detach().cpu().numpy())

    all_E = np.concatenate(all_E)
    fresh_E_mean = np.mean(all_E)
    fresh_E_std = np.std(all_E)

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
    }


def main():
    DIM = 24
    E_HARMONIC = DIM / 2.0  # 12.0 for 24D

    print("=" * 70)
    print(f"Dimensional Scaling Study Phase 2b: {DIM}D Henon-Heiles All-to-All")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    LAMBDA = 0.111803
    SEEDS = [42, 123, 456]

    # Run 3 independent trials
    print("\n" + "-" * 70)
    print(f"Running 3 independent trials (seeds: {SEEDS})")
    print("-" * 70)

    results = []

    for i, seed in enumerate(SEEDS):
        print(f"\n{'='*50}")
        print(f"TRIAL {i+1}/3 (seed={seed})")
        print(f"{'='*50}")

        result = run_single_trial(seed, dim=DIM, max_epochs=10000, max_time=45*60)
        results.append(result)

        print(f"\nTrial {i+1} complete:")
        print(f"  Final energy: {result['final_energy']:.6f}")
        print(f"  E_loc std: {result['final_std']:.4f}")
        print(f"  Fresh validation: {result['fresh_energy']:.6f} +/- {result['fresh_std']:.4f}")
        print(f"  Training time: {result['training_time']/60:.1f} min")
        print(f"  V<0 rejection: {result['v_rejection_rate']:.2%}")
        print(f"  Acceptance rate: {result['acceptance_rate']:.2%}")

    # Summary statistics
    print("\n" + "=" * 70)
    print(f"SUMMARY: {DIM}D All-to-All Henon-Heiles")
    print("=" * 70)

    final_energies = [r['final_energy'] for r in results]
    fresh_energies = [r['fresh_energy'] for r in results]
    final_stds = [r['final_std'] for r in results]
    times = [r['training_time'] for r in results]
    accept_rates = [r['acceptance_rate'] for r in results]
    v_reject_rates = [r['v_rejection_rate'] for r in results]

    mean_E = np.mean(final_energies)
    std_E = np.std(final_energies)
    cv_E = std_E / mean_E * 100

    mean_fresh = np.mean(fresh_energies)
    std_fresh = np.std(fresh_energies)

    print(f"\n| Metric                    | Value              |")
    print(f"|---------------------------|-------------------|")
    print(f"| Mean energy (training)    | {mean_E:.6f} +/- {std_E:.6f} |")
    print(f"| Mean energy (fresh valid) | {mean_fresh:.6f} +/- {std_fresh:.6f} |")
    print(f"| Coefficient of variation  | {cv_E:.4f}%          |")
    print(f"| Mean E_loc std            | {np.mean(final_stds):.4f}            |")
    print(f"| Mean training time        | {np.mean(times)/60:.1f} min          |")
    print(f"| Mean acceptance rate      | {np.mean(accept_rates):.2%}          |")
    print(f"| Mean V<0 rejection rate   | {np.mean(v_reject_rates):.2%}          |")

    print(f"\n| Run | Seed | Final E   | Fresh E   | E_loc std | Time  |")
    print(f"|-----|------|-----------|-----------|-----------|-------|")
    for i, r in enumerate(results):
        print(f"|  {i+1}  | {r['seed']:4d} | {r['final_energy']:.6f}  | {r['fresh_energy']:.6f}  | {r['final_std']:.4f}    | {r['training_time']/60:.1f}m |")

    # Scaling analysis
    print("\n" + "-" * 70)
    print("Extensive Scaling Analysis")
    print("-" * 70)

    E_per_dim = mean_E / DIM
    print(f"\nE_0/N = {mean_E:.6f} / {DIM} = {E_per_dim:.6f}")
    print(f"Expected (harmonic limit): 0.5")
    print(f"Deviation from harmonic: {(E_per_dim - 0.5) * 100:.2f}%")

    # Validation checks
    print("\n" + "-" * 70)
    print("Validation Checks")
    print("-" * 70)

    cv_ok = cv_E < 2.0  # Relaxed for higher D
    eloc_ok = np.mean(final_stds) < 2.0  # Relaxed for higher D
    scaling_ok = 0.45 < E_per_dim < 0.55
    fresh_ok = abs(mean_fresh - mean_E) / mean_E * 100 < 2.0  # Relaxed

    print(f"\n1. Self-consistency (CV < 2%):     {'PASS' if cv_ok else 'FAIL'} (CV = {cv_E:.4f}%)")
    print(f"2. E_loc variance (< 2.0):         {'PASS' if eloc_ok else 'FAIL'} (std = {np.mean(final_stds):.4f})")
    print(f"3. Extensive scaling (E/N ~ 0.5):  {'PASS' if scaling_ok else 'FAIL'} (E/N = {E_per_dim:.4f})")
    print(f"4. Fresh walker validation:        {'PASS' if fresh_ok else 'FAIL'} (diff = {abs(mean_fresh - mean_E)/mean_E*100:.3f}%)")

    all_pass = cv_ok and eloc_ok and scaling_ok and fresh_ok
    print(f"\nOVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

    # Save results
    results_dict = {
        'dim': DIM,
        'lambda': LAMBDA,
        'topology': 'all-to-all',
        'seeds': SEEDS,
        'mean_energy': mean_E,
        'std_energy': std_E,
        'cv_percent': cv_E,
        'E_per_dim': E_per_dim,
        'mean_fresh_energy': mean_fresh,
        'std_fresh_energy': std_fresh,
        'mean_eloc_std': np.mean(final_stds),
        'mean_time': np.mean(times),
        'mean_acceptance': np.mean(accept_rates),
        'mean_v_rejection': np.mean(v_reject_rates),
        'individual_results': results,
    }

    torch.save(results_dict, f'scaling_{DIM}d.pt')
    print(f"\nResults saved to: scaling_{DIM}d.pt")

    return results_dict


if __name__ == "__main__":
    results = main()
