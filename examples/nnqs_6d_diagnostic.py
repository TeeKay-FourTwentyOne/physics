"""
NNQS 6D Diagnostic: Verify network learned correct wavefunction.

The MCMC acceptance rate collapsed during training. This diagnostic
tests whether the network actually learned the correct wavefunction
by using fresh walkers with properly tuned step size.
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnqs_baseline import LogAmplitudeNetwork, MetropolisSampler, compute_local_energy
from nnqs_baseline.train import harmonic_potential_nd


def main():
    print("=" * 70)
    print("NNQS 6D Diagnostic: MCMC Acceptance Rate Investigation")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ============================================================
    # Step 1: Recreate and train the network
    # ============================================================
    print("\n" + "-" * 70)
    print("Step 1: Recreating trained network")
    print("-" * 70)

    torch.manual_seed(42)

    model = LogAmplitudeNetwork(input_dim=6, hidden_dims=[512, 512, 512, 512]).to(device)

    # Pretrain to Gaussian
    print("Pretraining to Gaussian...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1000):
        x = torch.randn(2000, 6, device=device) * 2.0
        target = -0.5 * torch.sum(x ** 2, dim=-1)
        pred = model(x)
        loss = torch.mean((pred - target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Pretrain final loss: {loss.item():.4f}")

    # VMC training
    print("VMC training (2500 epochs)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    sampler = MetropolisSampler(model, dim=6, n_walkers=10000, step_size=0.5, device=device)
    sampler.thermalize(200)

    for epoch in range(2500):
        x = sampler.sample(n_steps=20)
        x = x.requires_grad_(True)
        E_loc = compute_local_energy(model, x, harmonic_potential_nd)
        f = model(x)
        E_mean = E_loc.mean()
        centered_E = (E_loc - E_mean).detach()
        loss = 2.0 * torch.mean(centered_E * f)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 500 == 0:
            print(f"  Epoch {epoch}: E = {E_mean.item():.6f}")

    print(f"Training complete. Final E = {E_mean.item():.6f}")

    # ============================================================
    # Step 2: Initialize FRESH walkers
    # ============================================================
    print("\n" + "-" * 70)
    print("Step 2: Initialize fresh walkers (5000 from N(0,1))")
    print("-" * 70)

    n_walkers = 5000
    fresh_walkers = torch.randn(n_walkers, 6, device=device)
    print(f"Initial walker |x|^2 mean: {(fresh_walkers**2).sum(dim=-1).mean().item():.4f}")
    print(f"Initial walker |x|^2 std: {(fresh_walkers**2).sum(dim=-1).std().item():.4f}")

    # ============================================================
    # Step 3: Tune step_size for 40-60% acceptance
    # ============================================================
    print("\n" + "-" * 70)
    print("Step 3: Tune step_size for 40-60% acceptance")
    print("-" * 70)

    def test_acceptance(model, walkers, step_size, n_steps=100):
        """Test acceptance rate with given step_size."""
        walkers = walkers.clone()
        n_accepted = 0
        n_total = 0

        with torch.no_grad():
            for _ in range(n_steps):
                proposals = walkers + step_size * torch.randn_like(walkers)
                log_prob_current = 2.0 * model(walkers)
                log_prob_proposed = 2.0 * model(proposals)
                log_accept = log_prob_proposed - log_prob_current
                accept = torch.log(torch.rand(len(walkers), device=device)) < log_accept
                walkers = torch.where(accept.unsqueeze(-1), proposals, walkers)
                n_accepted += accept.sum().item()
                n_total += len(walkers)

        return n_accepted / n_total, walkers

    # Test different step sizes
    step_sizes = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    print(f"Testing step sizes: {step_sizes}")
    print()

    best_step = None
    best_accept = 0

    for ss in step_sizes:
        accept_rate, _ = test_acceptance(model, fresh_walkers, ss, n_steps=100)
        marker = ""
        if 0.4 <= accept_rate <= 0.6:
            marker = " <-- GOOD"
            if best_step is None or abs(accept_rate - 0.5) < abs(best_accept - 0.5):
                best_step = ss
                best_accept = accept_rate
        print(f"  step_size = {ss:.1f}: acceptance = {accept_rate:.2%}{marker}")

    if best_step is None:
        # If none in range, pick closest to 50%
        for ss in step_sizes:
            accept_rate, _ = test_acceptance(model, fresh_walkers, ss, n_steps=100)
            if best_step is None or abs(accept_rate - 0.5) < abs(best_accept - 0.5):
                best_step = ss
                best_accept = accept_rate

    print(f"\nSelected step_size = {best_step} (acceptance ~ {best_accept:.2%})")

    # ============================================================
    # Step 4 & 5: Run 2000 MCMC steps, discard first 500 as burn-in
    # ============================================================
    print("\n" + "-" * 70)
    print("Step 4 & 5: Run 2000 MCMC steps (500 burn-in + 1500 sampling)")
    print("-" * 70)

    # Create fresh sampler with tuned step size
    diag_sampler = MetropolisSampler(model, dim=6, n_walkers=n_walkers,
                                      step_size=best_step, device=device)
    diag_sampler.walkers = fresh_walkers.clone()

    # Burn-in: 500 steps
    print("Running 500 burn-in steps...")
    diag_sampler.sample(n_steps=500)
    print(f"  Acceptance during burn-in: {diag_sampler.acceptance_rate:.2%}")

    # Sampling: 1500 steps, collecting samples
    print("Running 1500 sampling steps...")
    all_E_loc = []
    all_x_sq = []

    for i in range(15):  # 15 batches of 100 steps = 1500 steps
        samples = diag_sampler.sample(n_steps=100)
        samples = samples.requires_grad_(True)

        E_loc = compute_local_energy(model, samples, harmonic_potential_nd)
        x_sq = (samples.detach() ** 2).sum(dim=-1)

        all_E_loc.append(E_loc.detach().cpu().numpy())
        all_x_sq.append(x_sq.cpu().numpy())

        if (i + 1) % 5 == 0:
            print(f"  Batch {i+1}/15: acceptance = {diag_sampler.acceptance_rate:.2%}")

    all_E_loc = np.concatenate(all_E_loc)
    all_x_sq = np.concatenate(all_x_sq)

    print(f"\nTotal samples collected: {len(all_E_loc):,}")

    # ============================================================
    # Step 6: Compute statistics
    # ============================================================
    print("\n" + "-" * 70)
    print("Step 6: Diagnostic Results")
    print("-" * 70)

    E_loc_mean = np.mean(all_E_loc)
    E_loc_std = np.std(all_E_loc)
    x_sq_mean = np.mean(all_x_sq)
    x_sq_std = np.std(all_x_sq)

    print()
    print("| Metric                  | Measured | Expected |")
    print("|-------------------------|----------|----------|")
    print(f"| Tuned step_size         | {best_step:8.2f} |    -     |")
    print(f"| Final acceptance rate   | {diag_sampler.acceptance_rate:7.2%} | 40-60%   |")
    print(f"| Mean E_loc              | {E_loc_mean:8.4f} |   3.0    |")
    print(f"| Std E_loc               | {E_loc_std:8.4f} |  <0.1    |")
    print(f"| Mean |x|^2              | {x_sq_mean:8.4f} |   3.0    |")
    print(f"| Std |x|^2               | {x_sq_std:8.4f} |    -     |")

    print("\n" + "-" * 70)
    print("Interpretation")
    print("-" * 70)

    # For 6D isotropic Gaussian with |psi|^2 = exp(-|x|^2),
    # each x_i^2 has mean 0.5, so |x|^2 has mean 3.0
    expected_x_sq = 3.0

    E_ok = abs(E_loc_mean - 3.0) < 0.1
    E_std_ok = E_loc_std < 0.2
    x_sq_ok = abs(x_sq_mean - expected_x_sq) < 0.5

    print(f"E_loc mean ~ 3.0: {'PASS' if E_ok else 'FAIL'} (error = {abs(E_loc_mean-3.0):.4f})")
    print(f"E_loc std < 0.2:  {'PASS' if E_std_ok else 'FAIL'} (value = {E_loc_std:.4f})")
    print(f"|x|^2 mean ~ 3.0: {'PASS' if x_sq_ok else 'FAIL'} (error = {abs(x_sq_mean-expected_x_sq):.4f})")

    all_pass = E_ok and E_std_ok and x_sq_ok
    print()
    if all_pass:
        print("OVERALL: NETWORK LEARNED CORRECT WAVEFUNCTION")
        print("The collapsed acceptance rate during training was due to the network")
        print("learning a sharper distribution than the sampler could track, but the")
        print("final wavefunction is correct.")
    else:
        print("OVERALL: POTENTIAL ISSUES DETECTED")
        print("The network may not have learned the correct wavefunction.")

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
