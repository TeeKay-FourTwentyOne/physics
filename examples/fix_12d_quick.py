"""
Quick fix test: 12D with 24D-style settings (smaller network).
"""

import torch
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnqs_baseline import LogAmplitudeNetwork, compute_local_energy


def hh_all_to_all(x, lam=0.111803):
    N = x.shape[1]
    V = 0.5 * torch.sum(x ** 2, dim=-1)
    for i in range(N):
        for j in range(i + 1, N):
            V = V + (lam / N) * (x[:, i] ** 2 * x[:, j] - x[:, j] ** 3 / 3)
    return V


def run_trial(seed, hidden_size, n_walkers, max_epochs=5000):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    DIM = 12
    V_func = lambda x: hh_all_to_all(x, 0.111803)

    model = LogAmplitudeNetwork(input_dim=DIM, hidden_dims=[hidden_size]*4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    walkers = 0.5 * torch.randn(n_walkers, DIM, device=device)
    step_size = 0.5

    start = time.time()
    for epoch in range(max_epochs):
        with torch.no_grad():
            for _ in range(20):
                prop = walkers + step_size * torch.randn_like(walkers)
                V_p = V_func(prop)
                v_ok = V_p >= 0
                lp_c = 2.0 * model(walkers)
                lp_p = 2.0 * model(prop)
                acc = v_ok & (torch.log(torch.rand(n_walkers, device=device)) < (lp_p - lp_c))
                walkers = torch.where(acc.unsqueeze(-1), prop, walkers)

        with torch.no_grad():
            V_s = V_func(walkers)
            hi = V_s > 32.0
            if hi.any():
                walkers[hi] = 0.3 * torch.randn(hi.sum().item(), DIM, device=device)

        x = walkers.clone().requires_grad_(True)
        E_loc = compute_local_energy(model, x, V_func)
        f = model(x)
        E_mean = E_loc.mean()

        loss = 2.0 * torch.mean((E_loc - E_mean).detach() * f)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 1000 == 0:
            print(f"  Epoch {epoch}: E = {E_mean.item():.4f}, std = {E_loc.std().item():.4f}")

    train_E = E_mean.item()
    train_time = time.time() - start

    # Fresh validation
    fresh = 0.5 * torch.randn(5000, DIM, device=device)
    with torch.no_grad():
        for _ in range(500):
            prop = fresh + 0.4 * torch.randn_like(fresh)
            V_p = V_func(prop)
            v_ok = V_p >= 0
            lp_c = 2.0 * model(fresh)
            lp_p = 2.0 * model(prop)
            acc = v_ok & (torch.log(torch.rand(5000, device=device)) < (lp_p - lp_c))
            fresh = torch.where(acc.unsqueeze(-1), prop, fresh)

    all_E = []
    for _ in range(10):
        with torch.no_grad():
            for _ in range(100):
                prop = fresh + 0.4 * torch.randn_like(fresh)
                V_p = V_func(prop)
                v_ok = V_p >= 0
                lp_c = 2.0 * model(fresh)
                lp_p = 2.0 * model(prop)
                acc = v_ok & (torch.log(torch.rand(5000, device=device)) < (lp_p - lp_c))
                fresh = torch.where(acc.unsqueeze(-1), prop, fresh)
        samples = fresh.clone().requires_grad_(True)
        E_loc = compute_local_energy(model, samples, V_func)
        all_E.append(E_loc.detach().cpu().numpy())

    all_E = np.concatenate(all_E)

    return {
        'train_E': train_E,
        'fresh_E': np.mean(all_E),
        'fresh_std': np.std(all_E),
        'time': train_time
    }


def main():
    print("=" * 70)
    print("12D FIX TEST: Comparing network sizes")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test with 24D-style settings (smaller network)
    print("\n--- Test 1: 24D-style settings (hidden=512, walkers=5000) ---")
    print("Running 3 seeds...")

    results_small = []
    for seed in [42, 123, 456]:
        print(f"\nSeed {seed}:")
        r = run_trial(seed, hidden_size=512, n_walkers=5000, max_epochs=5000)
        results_small.append(r)
        diff = abs(r['fresh_E'] - r['train_E']) / r['train_E'] * 100
        status = "OK" if r['fresh_std'] < 1.0 else "FAIL"
        print(f"  Train: {r['train_E']:.4f}, Fresh: {r['fresh_E']:.4f} +/- {r['fresh_std']:.4f}, Diff: {diff:.2f}% [{status}]")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n24D-style settings (hidden=512, walkers=5000):")
    print("| Seed | Train E | Fresh E | Fresh Std | Status |")
    print("|------|---------|---------|-----------|--------|")
    for i, r in enumerate(results_small):
        seed = [42, 123, 456][i]
        status = "OK" if r['fresh_std'] < 1.0 else "FAIL"
        print(f"| {seed:4d} | {r['train_E']:.4f}  | {r['fresh_E']:.4f}  | {r['fresh_std']:.4f}     | {status:6s} |")

    mean_fresh_std = np.mean([r['fresh_std'] for r in results_small])
    n_ok = sum(1 for r in results_small if r['fresh_std'] < 1.0)

    print(f"\nMean fresh std: {mean_fresh_std:.4f}")
    print(f"Trials passed: {n_ok}/3")

    if n_ok == 3:
        print("\nCONCLUSION: Smaller network FIXES the 12D problem!")
        print("Root cause: Original 12D network (896 hidden) was overparameterized.")
    else:
        print("\nCONCLUSION: Problem persists - need further investigation.")


if __name__ == "__main__":
    main()
