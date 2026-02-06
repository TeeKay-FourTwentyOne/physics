"""
Extended validation for 12D worst trial.
Retrain with seed=42 and do thorough fresh validation.
"""

import torch
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnqs_baseline import LogAmplitudeNetwork, compute_local_energy


def hh_all_to_all(x, lam=0.111803):
    N = x.shape[1]
    V = 0.5 * torch.sum(x ** 2, dim=-1)
    for i in range(N):
        for j in range(i + 1, N):
            V = V + (lam / N) * (x[:, i] ** 2 * x[:, j] - x[:, j] ** 3 / 3)
    return V


def main():
    print("=" * 70)
    print("EXTENDED 12D VALIDATION - WORST TRIAL (seed=42)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    DIM = 12
    LAMBDA = 0.111803
    V_func = lambda x: hh_all_to_all(x, LAMBDA)
    SEED = 42  # Worst trial

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Original 12D settings (896 hidden)
    hidden_size = 896
    model = LogAmplitudeNetwork(input_dim=DIM, hidden_dims=[hidden_size]*4).to(device)

    n_walkers = 10000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    walkers = 0.5 * torch.randn(n_walkers, DIM, device=device)
    step_size = 0.5

    print(f"\nTraining with original settings (hidden=896, walkers=10000)...")
    print(f"Running 5000 epochs...")

    start = time.time()
    for epoch in range(5000):
        with torch.no_grad():
            for _ in range(20):
                proposals = walkers + step_size * torch.randn_like(walkers)
                V_prop = V_func(proposals)
                v_valid = V_prop >= 0
                log_p_curr = 2.0 * model(walkers)
                log_p_prop = 2.0 * model(proposals)
                accept = v_valid & (torch.log(torch.rand(n_walkers, device=device)) < (log_p_prop - log_p_curr))
                walkers = torch.where(accept.unsqueeze(-1), proposals, walkers)

        with torch.no_grad():
            V_samples = V_func(walkers)
            too_high = V_samples > 32.0
            if too_high.any():
                walkers[too_high] = 0.3 * torch.randn(too_high.sum().item(), DIM, device=device)

        x = walkers.clone().requires_grad_(True)
        E_loc = compute_local_energy(model, x, V_func)
        f = model(x)
        E_mean = E_loc.mean()
        E_std = E_loc.std()

        centered_E = (E_loc - E_mean).detach()
        loss = 2.0 * torch.mean(centered_E * f)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Adaptive step
        if epoch > 0 and epoch % 200 == 0:
            with torch.no_grad():
                test_prop = walkers + step_size * torch.randn_like(walkers)
                V_test = V_func(test_prop)
                v_ok = V_test >= 0
                lp_c = 2.0 * model(walkers)
                lp_p = 2.0 * model(test_prop)
                acc = (v_ok & (torch.log(torch.rand(n_walkers, device=device)) < (lp_p - lp_c))).float().mean().item()
            if acc < 0.15:
                step_size *= 0.85
            elif acc > 0.50:
                step_size *= 1.15

        if epoch % 1000 == 0:
            print(f"  Epoch {epoch}: E = {E_mean.item():.4f}, std = {E_std.item():.4f}")

    train_time = time.time() - start
    final_train_E = E_mean.item()
    final_train_std = E_std.item()

    print(f"\nTraining complete in {train_time/60:.1f} min")
    print(f"Final training E = {final_train_E:.4f}, std = {final_train_std:.4f}")

    # ================================================================
    # Extended fresh validation
    # ================================================================
    print("\n" + "-" * 70)
    print("EXTENDED FRESH VALIDATION")
    print("-" * 70)
    print("10,000 fresh walkers, tuning step size, 5000 MCMC steps")

    fresh_walkers = 0.5 * torch.randn(10000, DIM, device=device)

    def mcmc_sample(walkers, ss, n_steps):
        walkers = walkers.clone()
        n_acc = 0
        with torch.no_grad():
            for _ in range(n_steps):
                prop = walkers + ss * torch.randn_like(walkers)
                V_p = V_func(prop)
                v_ok = V_p >= 0
                lp_c = 2.0 * model(walkers)
                lp_p = 2.0 * model(prop)
                acc = v_ok & (torch.log(torch.rand(len(walkers), device=device)) < (lp_p - lp_c))
                walkers = torch.where(acc.unsqueeze(-1), prop, walkers)
                n_acc += acc.sum().item()
        return n_acc / (n_steps * len(walkers)), walkers

    # Tune step size
    print("\nTuning step size for 30-50% acceptance:")
    best_ss = 0.4
    for ss in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        acc, _ = mcmc_sample(fresh_walkers[:2000], ss, 50)
        status = "<-- SELECTED" if 0.30 <= acc <= 0.50 and best_ss == 0.4 else ""
        print(f"  ss={ss:.2f}: acceptance={acc:.2%} {status}")
        if 0.30 <= acc <= 0.50:
            best_ss = ss
            break

    print(f"\nUsing step_size = {best_ss}")

    # Burn-in: 1000 steps
    print("Burn-in: 1000 steps...")
    _, fresh_walkers = mcmc_sample(fresh_walkers, best_ss, 1000)

    # Production: 4000 steps, collect every 100
    print("Production: 4000 steps...")
    all_E = []
    for batch in range(40):
        _, fresh_walkers = mcmc_sample(fresh_walkers, best_ss, 100)
        samples = fresh_walkers.clone().requires_grad_(True)
        E_loc = compute_local_energy(model, samples, V_func)
        all_E.append(E_loc.detach().cpu().numpy())
        if batch % 10 == 9:
            print(f"  Batch {batch+1}/40: E_mean = {np.mean(E_loc.detach().cpu().numpy()):.4f}")

    all_E = np.concatenate(all_E)

    final_acc, _ = mcmc_sample(fresh_walkers[:2000], best_ss, 100)

    print("\n" + "=" * 70)
    print("EXTENDED VALIDATION RESULTS")
    print("=" * 70)
    print(f"\nTraining E:      {final_train_E:.6f}")
    print(f"Fresh E mean:    {np.mean(all_E):.6f}")
    print(f"Fresh E std:     {np.std(all_E):.6f}")
    print(f"Fresh E min:     {np.min(all_E):.6f}")
    print(f"Fresh E max:     {np.max(all_E):.6f}")
    print(f"Acceptance rate: {final_acc:.2%}")
    print(f"\nDifference: {abs(np.mean(all_E) - final_train_E) / final_train_E * 100:.2f}%")

    # ================================================================
    # Network diagnostic
    # ================================================================
    print("\n" + "-" * 70)
    print("NETWORK DIAGNOSTIC")
    print("-" * 70)

    diag_walkers = fresh_walkers[:1000].clone()

    with torch.no_grad():
        x2_mean = torch.mean(torch.sum(diag_walkers ** 2, dim=-1)).item()
        V_mean = torch.mean(V_func(diag_walkers)).item()
        coord_stds = torch.std(diag_walkers, dim=0).cpu().numpy()
        x_max = torch.max(torch.abs(diag_walkers)).item()

    print(f"\nSample from |psi|^2:")
    print(f"  Mean |x|^2:    {x2_mean:.4f} (expected ~{DIM} for Gaussian)")
    print(f"  Mean V(x):     {V_mean:.4f} (expected ~E/2 ~ 3)")
    print(f"  Max |x|:       {x_max:.4f}")
    print(f"  Mean coord std: {np.mean(coord_stds):.4f} (expected ~1.0)")

    # Check for outliers in E_loc
    print(f"\nE_loc distribution:")
    percentiles = [1, 5, 25, 50, 75, 95, 99]
    for p in percentiles:
        print(f"  {p}th percentile: {np.percentile(all_E, p):.4f}")

    n_outliers = np.sum(np.abs(all_E - np.mean(all_E)) > 3 * np.std(all_E))
    print(f"  Outliers (>3 sigma): {n_outliers} / {len(all_E)} ({n_outliers/len(all_E)*100:.2f}%)")

    # ================================================================
    # Compare: retrain with 24D-style settings
    # ================================================================
    print("\n" + "=" * 70)
    print("COMPARISON: 12D WITH 24D-STYLE SETTINGS")
    print("=" * 70)
    print("hidden=512, walkers=5000")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model_small = LogAmplitudeNetwork(input_dim=DIM, hidden_dims=[512]*4).to(device)
    optimizer_small = torch.optim.Adam(model_small.parameters(), lr=1e-4)
    walkers_small = 0.5 * torch.randn(5000, DIM, device=device)
    step_size_small = 0.5

    print("\nTraining with 24D-style settings (5000 epochs)...")
    start = time.time()

    for epoch in range(5000):
        with torch.no_grad():
            for _ in range(20):
                prop = walkers_small + step_size_small * torch.randn_like(walkers_small)
                V_p = V_func(prop)
                v_ok = V_p >= 0
                lp_c = 2.0 * model_small(walkers_small)
                lp_p = 2.0 * model_small(prop)
                acc = v_ok & (torch.log(torch.rand(5000, device=device)) < (lp_p - lp_c))
                walkers_small = torch.where(acc.unsqueeze(-1), prop, walkers_small)

        with torch.no_grad():
            V_s = V_func(walkers_small)
            hi = V_s > 32.0
            if hi.any():
                walkers_small[hi] = 0.3 * torch.randn(hi.sum().item(), DIM, device=device)

        x = walkers_small.clone().requires_grad_(True)
        E_loc = compute_local_energy(model_small, x, V_func)
        f = model_small(x)
        E_mean = E_loc.mean()

        centered_E = (E_loc - E_mean).detach()
        loss = 2.0 * torch.mean(centered_E * f)

        optimizer_small.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_small.parameters(), 1.0)
        optimizer_small.step()

        if epoch % 1000 == 0:
            print(f"  Epoch {epoch}: E = {E_mean.item():.4f}")

    train_time_small = time.time() - start
    train_E_small = E_mean.item()

    # Fresh validation for small model
    print("\nFresh validation for small model...")
    fresh_small = 0.5 * torch.randn(5000, DIM, device=device)

    def mcmc_small(walkers, ss, n_steps):
        walkers = walkers.clone()
        with torch.no_grad():
            for _ in range(n_steps):
                prop = walkers + ss * torch.randn_like(walkers)
                V_p = V_func(prop)
                v_ok = V_p >= 0
                lp_c = 2.0 * model_small(walkers)
                lp_p = 2.0 * model_small(prop)
                acc = v_ok & (torch.log(torch.rand(len(walkers), device=device)) < (lp_p - lp_c))
                walkers = torch.where(acc.unsqueeze(-1), prop, walkers)
        return walkers

    fresh_small = mcmc_small(fresh_small, 0.4, 500)

    all_E_small = []
    for _ in range(10):
        fresh_small = mcmc_small(fresh_small, 0.4, 100)
        samples = fresh_small.clone().requires_grad_(True)
        E_loc = compute_local_energy(model_small, samples, V_func)
        all_E_small.append(E_loc.detach().cpu().numpy())

    all_E_small = np.concatenate(all_E_small)

    print(f"\n--- Small Model Results ---")
    print(f"Training E:   {train_E_small:.4f}")
    print(f"Fresh E mean: {np.mean(all_E_small):.4f}")
    print(f"Fresh E std:  {np.std(all_E_small):.4f}")
    print(f"Difference:   {abs(np.mean(all_E_small) - train_E_small) / train_E_small * 100:.2f}%")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
