"""
NNQS 6D Diagnostic v2: Use exact same training as Phase 2, then test with fresh walkers.
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
    print("NNQS 6D Diagnostic v2: Exact Phase 2 Training + Fresh Walker Test")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    # ============================================================
    # Step 1: EXACT same training as Phase 2
    # ============================================================
    print("\n" + "-" * 70)
    print("Step 1: Exact Phase 2 Training (with walker resetting)")
    print("-" * 70)

    model = LogAmplitudeNetwork(input_dim=6, hidden_dims=[512, 512, 512, 512]).to(device)

    # Pretrain to Gaussian
    print("Pretraining to Gaussian (1000 epochs)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1000):
        x = torch.randn(2000, 6, device=device) * 2.0
        target = -0.5 * torch.sum(x ** 2, dim=-1)
        pred = model(x)
        loss = torch.mean((pred - target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            print(f"  Pretrain {epoch}: loss = {loss.item():.4f}")

    # Test pretrained model
    print("\nTesting pretrained model (before VMC)...")
    test_x = torch.randn(5000, 6, device=device, requires_grad=True)
    E_loc_pretrain = compute_local_energy(model, test_x, harmonic_potential_nd)
    print(f"  E_loc mean after pretrain: {E_loc_pretrain.mean().item():.4f}")
    print(f"  E_loc std after pretrain: {E_loc_pretrain.std().item():.4f}")

    # VMC training with EXACT Phase 2 settings
    print("\nVMC training (2500 epochs with walker resetting)...")
    n_walkers = 10000
    mcmc_steps = 20
    max_sample_radius = 15.0
    lr = 5e-4
    grad_clip = 1.0

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sampler = MetropolisSampler(model, dim=6, n_walkers=n_walkers, step_size=0.5, device=device)
    sampler.thermalize(200)

    for epoch in range(2500):
        x = sampler.sample(n_steps=mcmc_steps)

        # CRITICAL: Reset walkers that drifted too far (same as Phase 2)
        sample_radius = torch.sqrt(torch.sum(x ** 2, dim=-1))
        too_far = sample_radius > max_sample_radius
        if too_far.any():
            n_reset = too_far.sum().item()
            sampler.walkers[too_far] = torch.randn(n_reset, 6, device=device)
            x = sampler.walkers.clone()

        x = x.requires_grad_(True)
        E_loc = compute_local_energy(model, x, harmonic_potential_nd)
        f = model(x)
        E_mean = E_loc.mean()
        centered_E = (E_loc - E_mean).detach()
        loss = 2.0 * torch.mean(centered_E * f)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if epoch % 500 == 0:
            print(f"  Epoch {epoch}: E = {E_mean.item():.6f}, accept = {sampler.acceptance_rate:.2%}")

    print(f"\nTraining complete. Final E = {E_mean.item():.6f}")

    # ============================================================
    # Step 2: Test with FRESH walkers (the real test)
    # ============================================================
    print("\n" + "-" * 70)
    print("Step 2: Test with FRESH walkers")
    print("-" * 70)

    # Initialize completely fresh walkers from N(0,1)
    n_test_walkers = 5000
    fresh_walkers = torch.randn(n_test_walkers, 6, device=device)

    # Find good step size
    def test_acceptance(walkers, step_size, n_steps=100):
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

    print("Tuning step_size...")
    step_sizes = [0.3, 0.5, 0.7, 1.0, 1.5]
    best_step = 0.7
    best_accept_diff = float('inf')

    for ss in step_sizes:
        acc, _ = test_acceptance(fresh_walkers, ss)
        diff = abs(acc - 0.5)
        print(f"  step_size={ss}: acceptance={acc:.2%}")
        if diff < best_accept_diff:
            best_accept_diff = diff
            best_step = ss

    print(f"Using step_size = {best_step}")

    # Run MCMC with fresh walkers
    print("\nRunning MCMC: 500 burn-in + 1500 sampling steps...")
    walkers = fresh_walkers.clone()

    # Burn-in
    for _ in range(500):
        proposals = walkers + best_step * torch.randn_like(walkers)
        with torch.no_grad():
            log_prob_current = 2.0 * model(walkers)
            log_prob_proposed = 2.0 * model(proposals)
        log_accept = log_prob_proposed - log_prob_current
        accept = torch.log(torch.rand(n_test_walkers, device=device)) < log_accept
        walkers = torch.where(accept.unsqueeze(-1), proposals, walkers)

    # Sampling
    all_E_loc = []
    all_x_sq = []
    n_accepted = 0
    n_total = 0

    for step in range(1500):
        proposals = walkers + best_step * torch.randn_like(walkers)
        with torch.no_grad():
            log_prob_current = 2.0 * model(walkers)
            log_prob_proposed = 2.0 * model(proposals)
        log_accept = log_prob_proposed - log_prob_current
        accept = torch.log(torch.rand(n_test_walkers, device=device)) < log_accept
        walkers = torch.where(accept.unsqueeze(-1), proposals, walkers)
        n_accepted += accept.sum().item()
        n_total += n_test_walkers

        # Compute E_loc every 100 steps
        if step % 100 == 0:
            w_grad = walkers.clone().requires_grad_(True)
            E_loc = compute_local_energy(model, w_grad, harmonic_potential_nd)
            x_sq = (walkers ** 2).sum(dim=-1)
            all_E_loc.append(E_loc.detach().cpu().numpy())
            all_x_sq.append(x_sq.cpu().numpy())

    all_E_loc = np.concatenate(all_E_loc)
    all_x_sq = np.concatenate(all_x_sq)
    final_accept = n_accepted / n_total

    # ============================================================
    # Results
    # ============================================================
    print("\n" + "-" * 70)
    print("DIAGNOSTIC RESULTS")
    print("-" * 70)

    E_loc_mean = np.mean(all_E_loc)
    E_loc_std = np.std(all_E_loc)
    x_sq_mean = np.mean(all_x_sq)
    x_sq_std = np.std(all_x_sq)

    print()
    print("| Metric                  | Measured   | Expected |")
    print("|-------------------------|------------|----------|")
    print(f"| Final acceptance rate   | {final_accept:9.2%} | 40-60%   |")
    print(f"| Mean E_loc              | {E_loc_mean:10.4f} |   3.0    |")
    print(f"| Std E_loc               | {E_loc_std:10.4f} |  <0.1    |")
    print(f"| Mean |x|^2              | {x_sq_mean:10.4f} |   3.0    |")
    print(f"| Std |x|^2               | {x_sq_std:10.4f} |  ~2.4    |")

    # For chi-squared(6) / 2, mean = 3, var = 6, std = 2.45
    expected_x_sq_std = np.sqrt(6)

    print("\n" + "-" * 70)
    print("Interpretation")
    print("-" * 70)

    E_ok = abs(E_loc_mean - 3.0) < 0.2
    E_std_ok = E_loc_std < 0.3
    x_sq_ok = abs(x_sq_mean - 3.0) < 0.5

    print(f"E_loc mean ~ 3.0: {'PASS' if E_ok else 'FAIL'} (error = {abs(E_loc_mean-3.0):.4f})")
    print(f"E_loc std < 0.3:  {'PASS' if E_std_ok else 'FAIL'} (value = {E_loc_std:.4f})")
    print(f"|x|^2 mean ~ 3.0: {'PASS' if x_sq_ok else 'FAIL'} (error = {abs(x_sq_mean-3.0):.4f})")

    all_pass = E_ok and E_std_ok and x_sq_ok
    print()
    if all_pass:
        print("OVERALL: NETWORK LEARNED CORRECT WAVEFUNCTION")
    else:
        print("OVERALL: NETWORK DID NOT LEARN CORRECT WAVEFUNCTION")
        print("\nThe Phase 2 'success' was likely due to pretraining giving a good")
        print("initial approximation, but VMC training did not improve it (and may")
        print("have made it worse). The collapsed MCMC acceptance rate hid this issue.")

    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
