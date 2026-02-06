"""
Investigate 12D Fresh Validation Anomaly

Diagnose why 12D showed training E=6.08 but fresh validation E=6.46 with high variance.
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnqs_baseline import LogAmplitudeNetwork, compute_local_energy


def hh_all_to_all(x, lam=0.111803):
    """N-dimensional Henon-Heiles with all-to-all coupling."""
    N = x.shape[1]
    V = 0.5 * torch.sum(x ** 2, dim=-1)
    for i in range(N):
        for j in range(i + 1, N):
            V = V + (lam / N) * (x[:, i] ** 2 * x[:, j] - x[:, j] ** 3 / 3)
    return V


def main():
    print("=" * 70)
    print("INVESTIGATING 12D FRESH VALIDATION ANOMALY")
    print("=" * 70)

    # Load 12D results
    data = torch.load('scaling_12d.pt', weights_only=False)

    # ================================================================
    # 1. Show individual fresh validation results
    # ================================================================
    print("\n" + "-" * 70)
    print("1. INDIVIDUAL TRIAL FRESH VALIDATION RESULTS")
    print("-" * 70)

    print(f"\n| Trial | Seed | Training E | Fresh E    | Fresh Std  | Diff (%) |")
    print(f"|-------|------|------------|------------|------------|----------|")

    worst_trial = None
    worst_diff = 0

    for i, r in enumerate(data['individual_results']):
        train_E = r['final_energy']
        fresh_E = r['fresh_energy']
        fresh_std = r['fresh_std']
        diff_pct = abs(fresh_E - train_E) / train_E * 100

        print(f"|   {i+1}   | {r['seed']:4d} | {train_E:10.4f} | {fresh_E:10.4f} | {fresh_std:10.4f} | {diff_pct:8.2f} |")

        if diff_pct > worst_diff:
            worst_diff = diff_pct
            worst_trial = i
            worst_seed = r['seed']

    print(f"\nWorst trial: #{worst_trial + 1} (seed={worst_seed}) with {worst_diff:.2f}% difference")
    print(f"Fresh std for trials: {[r['fresh_std'] for r in data['individual_results']]}")

    # Identify problematic trials (high fresh_std indicates sampling issues)
    print("\nAnalysis:")
    for i, r in enumerate(data['individual_results']):
        if r['fresh_std'] > 1.0:
            print(f"  Trial {i+1}: HUGE fresh_std = {r['fresh_std']:.2f} - likely sampling problem")
        elif r['fresh_std'] > 0.5:
            print(f"  Trial {i+1}: High fresh_std = {r['fresh_std']:.2f} - may have issues")
        else:
            print(f"  Trial {i+1}: Normal fresh_std = {r['fresh_std']:.2f}")

    # ================================================================
    # 2. Retrain worst trial and do extended fresh validation
    # ================================================================
    print("\n" + "-" * 70)
    print("2. EXTENDED FRESH VALIDATION FOR WORST TRIAL")
    print("-" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Retrain the worst trial to get the model
    print(f"\nRetraining trial {worst_trial + 1} (seed={worst_seed}) to get model...")

    torch.manual_seed(worst_seed)
    np.random.seed(worst_seed)

    DIM = 12
    LAMBDA = 0.111803
    V_func = lambda x: hh_all_to_all(x, LAMBDA)

    # Same network architecture as 12D script
    hidden_size = 512 + 64 * (DIM - 6)  # 896 for 12D
    model = LogAmplitudeNetwork(input_dim=DIM, hidden_dims=[hidden_size, hidden_size, hidden_size, hidden_size]).to(device)

    # Quick retrain (abbreviated - just enough to reproduce the issue)
    n_walkers = 10000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Initialize walkers
    walkers = 0.5 * torch.randn(n_walkers, DIM, device=device)
    step_size = 0.5

    print("Training (abbreviated, 3000 epochs)...")
    for epoch in range(3000):
        # MCMC sampling
        with torch.no_grad():
            for _ in range(20):
                proposals = walkers + step_size * torch.randn_like(walkers)
                V_prop = V_func(proposals)
                v_valid = V_prop >= 0
                log_p_curr = 2.0 * model(walkers)
                log_p_prop = 2.0 * model(proposals)
                accept = v_valid & (torch.log(torch.rand(n_walkers, device=device)) < (log_p_prop - log_p_curr))
                walkers = torch.where(accept.unsqueeze(-1), proposals, walkers)

        # Reset high-V walkers
        with torch.no_grad():
            V_samples = V_func(walkers)
            too_high = V_samples > 32.0
            if too_high.any():
                walkers[too_high] = 0.3 * torch.randn(too_high.sum().item(), DIM, device=device)

        x = walkers.clone().requires_grad_(True)
        E_loc = compute_local_energy(model, x, V_func)
        f = model(x)
        E_mean = E_loc.mean()

        centered_E = (E_loc - E_mean).detach()
        loss = 2.0 * torch.mean(centered_E * f)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 500 == 0:
            print(f"  Epoch {epoch}: E = {E_mean.item():.4f}, std = {E_loc.std().item():.4f}")

    final_train_E = E_mean.item()
    print(f"\nFinal training energy: {final_train_E:.4f}")

    # Extended fresh validation
    print("\n--- Extended Fresh Validation ---")
    print("Using 10,000 fresh walkers, tuning step size...")

    fresh_walkers = 0.5 * torch.randn(10000, DIM, device=device)

    # Tune step size to get 30-50% acceptance
    def test_acceptance(walkers, step_size, n_steps=100):
        walkers = walkers.clone()
        n_acc = 0
        with torch.no_grad():
            for _ in range(n_steps):
                proposals = walkers + step_size * torch.randn_like(walkers)
                V_prop = V_func(proposals)
                v_valid = V_prop >= 0
                log_p_curr = 2.0 * model(walkers)
                log_p_prop = 2.0 * model(proposals)
                accept = v_valid & (torch.log(torch.rand(len(walkers), device=device)) < (log_p_prop - log_p_curr))
                walkers = torch.where(accept.unsqueeze(-1), proposals, walkers)
                n_acc += accept.sum().item()
        return n_acc / (n_steps * len(walkers)), walkers

    print("\nTuning step size:")
    best_ss = 0.5
    for ss in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        acc, _ = test_acceptance(fresh_walkers[:1000], ss, n_steps=50)
        print(f"  step_size={ss:.1f}: acceptance={acc:.2%}")
        if 0.30 < acc < 0.50:
            best_ss = ss
            break
        elif acc > 0.50 and ss > best_ss:
            best_ss = ss

    print(f"\nUsing step_size={best_ss}")

    # Run 5000 MCMC steps, discard first 1000 as burn-in
    print("Running 5000 MCMC steps (1000 burn-in + 4000 production)...")

    fresh_walkers = 0.5 * torch.randn(10000, DIM, device=device)

    # Burn-in
    _, fresh_walkers = test_acceptance(fresh_walkers, best_ss, n_steps=1000)

    # Production - collect E_loc every 100 steps
    all_E = []
    for batch in range(40):  # 40 batches of 100 steps = 4000 steps
        _, fresh_walkers = test_acceptance(fresh_walkers, best_ss, n_steps=100)
        samples = fresh_walkers.clone().requires_grad_(True)
        E_loc = compute_local_energy(model, samples, V_func)
        all_E.append(E_loc.detach().cpu().numpy())
        if batch % 10 == 0:
            print(f"  Batch {batch+1}/40: E_loc mean = {np.mean(E_loc.detach().cpu().numpy()):.4f}")

    all_E = np.concatenate(all_E)

    print(f"\n--- Extended Fresh Validation Results ---")
    print(f"E_loc mean: {np.mean(all_E):.6f}")
    print(f"E_loc std:  {np.std(all_E):.6f}")
    print(f"E_loc min:  {np.min(all_E):.6f}")
    print(f"E_loc max:  {np.max(all_E):.6f}")

    final_accept, _ = test_acceptance(fresh_walkers[:1000], best_ss, n_steps=100)
    print(f"Final acceptance rate: {final_accept:.2%}")

    # ================================================================
    # 3. Check if network learned something wrong
    # ================================================================
    print("\n" + "-" * 70)
    print("3. NETWORK DIAGNOSTIC: What did the model learn?")
    print("-" * 70)

    # Sample 1000 points from trained network's |psi|^2
    print("\nSampling 1000 points from |psi|^2...")

    diag_walkers = 0.5 * torch.randn(1000, DIM, device=device)
    _, diag_walkers = test_acceptance(diag_walkers, best_ss, n_steps=500)  # thermalize

    with torch.no_grad():
        # Mean |x|^2
        x2_mean = torch.mean(torch.sum(diag_walkers ** 2, dim=-1)).item()

        # Mean V(x)
        V_mean = torch.mean(V_func(diag_walkers)).item()

        # Per-coordinate statistics
        coord_means = torch.mean(diag_walkers, dim=0).cpu().numpy()
        coord_stds = torch.std(diag_walkers, dim=0).cpu().numpy()

        # Check for any coordinates with anomalous values
        x_max = torch.max(torch.abs(diag_walkers)).item()

    print(f"\nSample statistics:")
    print(f"  Mean |x|^2: {x2_mean:.4f} (expected ~{DIM} for Gaussian-like ground state)")
    print(f"  Mean V(x):  {V_mean:.4f} (expected ~E/2 = ~3 for virial theorem)")
    print(f"  Max |x|:    {x_max:.4f}")

    print(f"\nPer-coordinate means: {coord_means}")
    print(f"Per-coordinate stds:  {coord_stds}")

    # Check for distribution issues
    print(f"\nExpected per-coordinate std for Gaussian: ~1.0")
    print(f"Actual mean std: {np.mean(coord_stds):.4f}")

    if np.mean(coord_stds) > 1.5:
        print("WARNING: Wavefunction is too spread out!")
    elif np.mean(coord_stds) < 0.5:
        print("WARNING: Wavefunction is too localized!")
    else:
        print("OK: Wavefunction spread looks reasonable")

    # ================================================================
    # 4. Compare 12D vs 24D hyperparameters
    # ================================================================
    print("\n" + "-" * 70)
    print("4. HYPERPARAMETER COMPARISON: 12D vs 24D")
    print("-" * 70)

    print(f"""
| Parameter      | 12D Script         | 24D Script         |
|----------------|--------------------|--------------------|
| hidden_size    | 896 (512+64*6)     | 512 (capped)       |
| n_walkers      | 10,000             | 5,000              |
| max_V_reset    | 32.0               | 56.0               |
| Network params | ~3.2M              | ~1.1M              |

Key differences:
1. 12D used LARGER network (896 vs 512 hidden)
2. 12D used MORE walkers (10k vs 5k)
3. 24D capped hidden size at 512 for memory

Hypothesis: 12D may have been overfit or undertrained relative to its capacity.
The larger network (896) may need more training to converge properly.
24D's smaller network (512) was a better match for the training budget.
""")

    # ================================================================
    # 5. Quick comparison: retrain 12D with 24D settings
    # ================================================================
    print("\n" + "-" * 70)
    print("5. QUICK TEST: 12D with 24D-style settings")
    print("-" * 70)

    print("Retraining 12D with smaller network (512) and fewer walkers (5000)...")

    torch.manual_seed(42)
    np.random.seed(42)

    # 24D-style settings
    model_small = LogAmplitudeNetwork(input_dim=DIM, hidden_dims=[512, 512, 512, 512]).to(device)
    optimizer_small = torch.optim.Adam(model_small.parameters(), lr=1e-4)

    walkers_small = 0.5 * torch.randn(5000, DIM, device=device)
    step_size = 0.5

    for epoch in range(3000):
        with torch.no_grad():
            for _ in range(20):
                proposals = walkers_small + step_size * torch.randn_like(walkers_small)
                V_prop = V_func(proposals)
                v_valid = V_prop >= 0
                log_p_curr = 2.0 * model_small(walkers_small)
                log_p_prop = 2.0 * model_small(proposals)
                accept = v_valid & (torch.log(torch.rand(5000, device=device)) < (log_p_prop - log_p_curr))
                walkers_small = torch.where(accept.unsqueeze(-1), proposals, walkers_small)

        with torch.no_grad():
            V_samples = V_func(walkers_small)
            too_high = V_samples > 32.0
            if too_high.any():
                walkers_small[too_high] = 0.3 * torch.randn(too_high.sum().item(), DIM, device=device)

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

        if epoch % 500 == 0:
            print(f"  Epoch {epoch}: E = {E_mean.item():.4f}, std = {E_loc.std().item():.4f}")

    # Fresh validation with small model
    print("\nFresh validation for small model...")
    fresh_small = 0.5 * torch.randn(5000, DIM, device=device)

    def test_small(walkers, ss, n_steps):
        walkers = walkers.clone()
        n_acc = 0
        with torch.no_grad():
            for _ in range(n_steps):
                proposals = walkers + ss * torch.randn_like(walkers)
                V_prop = V_func(proposals)
                v_valid = V_prop >= 0
                log_p_curr = 2.0 * model_small(walkers)
                log_p_prop = 2.0 * model_small(proposals)
                accept = v_valid & (torch.log(torch.rand(len(walkers), device=device)) < (log_p_prop - log_p_curr))
                walkers = torch.where(accept.unsqueeze(-1), proposals, walkers)
                n_acc += accept.sum().item()
        return n_acc / (n_steps * len(walkers)), walkers

    _, fresh_small = test_small(fresh_small, 0.4, 500)  # burn-in

    all_E_small = []
    for _ in range(10):
        _, fresh_small = test_small(fresh_small, 0.4, 100)
        samples = fresh_small.clone().requires_grad_(True)
        E_loc = compute_local_energy(model_small, samples, V_func)
        all_E_small.append(E_loc.detach().cpu().numpy())

    all_E_small = np.concatenate(all_E_small)

    print(f"\n--- Small Model Fresh Validation ---")
    print(f"Training E:    {E_mean.item():.4f}")
    print(f"Fresh E mean:  {np.mean(all_E_small):.4f}")
    print(f"Fresh E std:   {np.std(all_E_small):.4f}")
    print(f"Difference:    {abs(np.mean(all_E_small) - E_mean.item()) / E_mean.item() * 100:.2f}%")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)


if __name__ == "__main__":
    main()
