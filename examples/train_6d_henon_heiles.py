"""
NNQS Phase 4: 6D Henon-Heiles Potential

System (standard benchmark parameters):
    V(x) = 1/2 * sum(x_i^2) + lambda * sum(x_i^2 * x_{i+1} - x_{i+1}^3/3)
    lambda = 0.111803 (literature standard)
    E_exact ~ 2.99 a.u. (Nest & Meyer 2002)

Key challenges:
    - Cubic terms make this genuinely anharmonic
    - Potential is UNBOUNDED in certain directions
    - Ground state is NOT Gaussian in any coordinate system
"""

import torch
import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nnqs_baseline import LogAmplitudeNetwork, MetropolisSampler, compute_local_energy


def henon_heiles_potential(x, lam=0.111803):
    """
    6D Henon-Heiles potential with periodic boundary.

    V(x) = 1/2 * sum(x_i^2) + lambda * sum(x_i^2 * x_{i+1} - x_{i+1}^3/3)

    Args:
        x: positions, shape (batch, 6)
        lam: coupling strength (0.111803 is literature standard)

    Returns:
        V: potential energy, shape (batch,)
    """
    # Harmonic part: 1/2 * sum(x_i^2)
    V = 0.5 * torch.sum(x ** 2, dim=-1)

    # Anharmonic coupling (periodic: x_7 = x_1)
    for i in range(6):
        j = (i + 1) % 6
        V = V + lam * (x[:, i] ** 2 * x[:, j] - x[:, j] ** 3 / 3)

    return V


def main():
    print("=" * 70)
    print("NNQS Phase 4: 6D Henon-Heiles Potential")
    print("V(x) = 0.5*sum(x_i^2) + lambda*sum(x_i^2*x_{i+1} - x_{i+1}^3/3)")
    print("lambda = 0.111803, E_exact ~ 2.99 a.u.")
    print("=" * 70)

    E_EXPECTED = 2.99  # Benchmark from Nest & Meyer 2002
    LAMBDA = 0.111803
    MAX_EPOCHS = 10000
    MAX_TIME = 60 * 60  # 60 minutes
    CHECKPOINT_INTERVAL = 1000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    torch.manual_seed(42)
    np.random.seed(42)

    # ============================================================
    # Initialize network (random weights, no pretraining)
    # ============================================================
    print("\n" + "-" * 70)
    print("Initializing network (RANDOM weights, no pretraining)")
    print("-" * 70)

    model = LogAmplitudeNetwork(input_dim=6, hidden_dims=[512, 512, 512, 512]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Network parameters: {n_params:,}")

    # Check initial state
    test_x = torch.randn(1000, 6, device=device, requires_grad=True)
    E_loc_init = compute_local_energy(model, test_x, lambda x: henon_heiles_potential(x, LAMBDA))
    V_init = henon_heiles_potential(test_x.detach(), LAMBDA)
    print(f"Initial E_loc mean: {E_loc_init.mean().item():.4f}")
    print(f"Initial E_loc std: {E_loc_init.std().item():.4f}")
    print(f"Initial V mean: {V_init.mean().item():.4f}")

    # ============================================================
    # VMC Training
    # ============================================================
    print("\n" + "-" * 70)
    print("VMC Training")
    print("-" * 70)

    n_walkers = 10000
    mcmc_steps = 20
    max_sample_radius = 20.0  # Increased for wider potential well
    lr = 1e-4  # Gentler learning rate for anharmonic potential
    grad_clip = 1.0
    target_accept_low = 0.30
    target_accept_high = 0.50

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sampler = MetropolisSampler(model, dim=6, n_walkers=n_walkers, step_size=0.5, device=device)

    print("Thermalizing sampler...")
    sampler.thermalize(200)

    # Training history
    energies = []
    energy_stds = []
    acceptance_rates = []
    max_V_history = []
    escaped_walkers_history = []

    best_energy = float('inf')
    best_error = float('inf')
    best_epoch = 0
    epoch_at_1pct = None
    epoch_at_5pct = None

    start_time = time.time()

    print(f"\nTarget: E ~ {E_EXPECTED} a.u. (within 1%: {E_EXPECTED*0.99:.4f} - {E_EXPECTED*1.01:.4f})")
    print(f"\n{'Epoch':>6} {'Energy':>10} {'E_std':>8} {'Accept':>8} {'Error%':>8} {'MaxV':>8} {'Escaped':>8} {'Time':>8}")
    print("-" * 80)

    for epoch in range(MAX_EPOCHS):
        elapsed = time.time() - start_time
        if elapsed > MAX_TIME:
            print(f"\nReached max time ({MAX_TIME/60:.0f} minutes)")
            break

        # Sample
        x = sampler.sample(n_steps=mcmc_steps)

        # Check for walkers in high-V regions (potential escape)
        with torch.no_grad():
            V_samples = henon_heiles_potential(x, LAMBDA)
            max_V = V_samples.max().item()
            n_escaped = (V_samples > 50).sum().item()  # V > 50 is definitely escaped

        # Reset drifted walkers (radius OR high potential)
        sample_radius = torch.sqrt(torch.sum(x ** 2, dim=-1))
        too_far = (sample_radius > max_sample_radius) | (V_samples > 30)
        if too_far.any():
            n_reset = too_far.sum().item()
            # Reset to positions near origin
            sampler.walkers[too_far] = 0.5 * torch.randn(n_reset, 6, device=device)
            x = sampler.walkers.clone()

        x = x.requires_grad_(True)

        # Compute local energy
        E_loc = compute_local_energy(model, x, lambda x: henon_heiles_potential(x, LAMBDA))
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
        max_V_history.append(max_V)
        escaped_walkers_history.append(n_escaped)

        error_pct = abs(E_val - E_EXPECTED) / E_EXPECTED * 100

        if error_pct < best_error:
            best_error = error_pct
            best_energy = E_val
            best_epoch = epoch

        # Track milestones
        if epoch_at_5pct is None and error_pct < 5.0:
            epoch_at_5pct = epoch
            mins = elapsed / 60
            print(f"\n*** Reached 5% error at epoch {epoch} ({mins:.1f} min)! ***\n")

        if epoch_at_1pct is None and error_pct < 1.0:
            epoch_at_1pct = epoch
            mins = elapsed / 60
            print(f"\n*** Reached 1% error at epoch {epoch} ({mins:.1f} min)! ***\n")

        # Print every 100 epochs
        if epoch % 100 == 0:
            mins = elapsed / 60
            print(f"{epoch:6d} {E_val:10.4f} {E_std.item():8.4f} "
                  f"{sampler.acceptance_rate:7.2%} {error_pct:8.2f}% {max_V:8.1f} {n_escaped:8d} {mins:7.1f}m")

        # Adaptive step size (target 30-50%)
        if epoch > 0 and epoch % 200 == 0:
            avg_accept = np.mean(acceptance_rates[-200:])
            if avg_accept < target_accept_low * 0.8:
                sampler.step_size *= 0.85
                print(f"  [Step size -> {sampler.step_size:.3f}]")
            elif avg_accept > target_accept_high * 1.2:
                sampler.step_size *= 1.15
                print(f"  [Step size -> {sampler.step_size:.3f}]")

        # Checkpoint
        if epoch > 0 and epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'energies': energies,
                'energy_stds': energy_stds,
                'acceptance_rates': acceptance_rates,
                'best_energy': best_energy,
                'best_error': best_error,
            }
            torch.save(checkpoint, 'nnqs_6d_hh_checkpoint.pt')
            print(f"  [Checkpoint saved at epoch {epoch}]")

    total_time = time.time() - start_time
    final_epochs = len(energies)

    # Save final checkpoint
    torch.save({
        'epoch': final_epochs,
        'model_state_dict': model.state_dict(),
        'energies': energies,
        'energy_stds': energy_stds,
        'acceptance_rates': acceptance_rates,
        'best_energy': best_energy,
        'best_error': best_error,
    }, 'nnqs_6d_hh_checkpoint.pt')

    # ============================================================
    # Results
    # ============================================================
    print("\n" + "=" * 70)
    print("TRAINING RESULTS")
    print("=" * 70)

    final_E = energies[-1]
    final_std = energy_stds[-1]
    final_error = abs(final_E - E_EXPECTED) / E_EXPECTED * 100
    avg_accept = np.mean(acceptance_rates[-500:]) if len(acceptance_rates) >= 500 else np.mean(acceptance_rates)

    print(f"\n| Metric                     | Value           |")
    print(f"|----------------------------|-----------------|")
    print(f"| Final energy               | {final_E:15.6f} |")
    print(f"| Best energy                | {best_energy:15.6f} |")
    print(f"| Expected energy            | {E_EXPECTED:15.6f} |")
    print(f"| Final error                | {final_error:14.2f}% |")
    print(f"| Best error                 | {best_error:14.2f}% |")
    print(f"| Best error at epoch        | {best_epoch:15d} |")
    print(f"| Final E_loc std            | {final_std:15.4f} |")
    print(f"| Total epochs               | {final_epochs:15d} |")
    print(f"| Training time              | {total_time/60:13.1f} min |")
    print(f"| Epochs to 1% error         | {str(epoch_at_1pct) if epoch_at_1pct else 'N/A':>15} |")
    print(f"| Avg acceptance (last 500)  | {avg_accept:14.2%} |")

    # ============================================================
    # Fresh Walker Validation
    # ============================================================
    print("\n" + "-" * 70)
    print("Fresh Walker Validation")
    print("-" * 70)

    fresh_walkers = torch.randn(5000, 6, device=device) * 0.7  # Start closer to origin

    def test_acceptance_hh(walkers, step_size, n_steps=100):
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
    print("Tuning step size for fresh walkers...")
    best_ss = 0.5
    best_diff = 1.0
    for ss in [0.2, 0.3, 0.4, 0.5, 0.7]:
        acc, _ = test_acceptance_hh(fresh_walkers, ss)
        diff = abs(acc - 0.4)
        print(f"  step_size={ss}: acceptance={acc:.2%}")
        if diff < best_diff:
            best_diff = diff
            best_ss = ss

    print(f"Using step_size = {best_ss}")

    # Burn-in
    print("Running burn-in (500 steps)...")
    _, walkers = test_acceptance_hh(fresh_walkers, best_ss, n_steps=500)

    # Sample
    print("Sampling (1500 steps)...")
    all_E = []
    all_V = []
    all_x_sq = []
    for i in range(15):
        acc, walkers = test_acceptance_hh(walkers, best_ss, n_steps=100)
        w = walkers.clone().requires_grad_(True)
        E_loc = compute_local_energy(model, w, lambda x: henon_heiles_potential(x, LAMBDA))
        V = henon_heiles_potential(walkers, LAMBDA)
        all_E.append(E_loc.detach().cpu().numpy())
        all_V.append(V.cpu().numpy())
        all_x_sq.append((walkers ** 2).sum(dim=-1).cpu().numpy())

    all_E = np.concatenate(all_E)
    all_V = np.concatenate(all_V)
    all_x_sq = np.concatenate(all_x_sq)

    fresh_E_mean = np.mean(all_E)
    fresh_E_std = np.std(all_E)
    fresh_V_mean = np.mean(all_V)
    fresh_V_max = np.max(all_V)
    n_in_well = np.sum(all_V < 10)
    pct_in_well = n_in_well / len(all_V) * 100

    print(f"\n| Metric                  | Measured   | Expected   |")
    print(f"|-------------------------|------------|------------|")
    print(f"| E_loc mean              | {fresh_E_mean:10.4f} | {E_EXPECTED:10.4f} |")
    print(f"| E_loc std               | {fresh_E_std:10.4f} |    <0.5    |")
    print(f"| V mean                  | {fresh_V_mean:10.4f} |    ~1-2    |")
    print(f"| V max                   | {fresh_V_max:10.4f} |    <10     |")
    print(f"| % walkers in well (V<10)| {pct_in_well:9.1f}% |   >95%     |")

    fresh_error = abs(fresh_E_mean - E_EXPECTED) / E_EXPECTED * 100
    print(f"\nFresh walker energy error: {fresh_error:.2f}%")

    # ============================================================
    # Convergence Plot
    # ============================================================
    print("\n" + "-" * 70)
    print("Generating convergence plot...")
    print("-" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    epochs_arr = np.arange(len(energies))

    # Energy
    ax1 = axes[0, 0]
    ax1.plot(epochs_arr, energies, 'b-', alpha=0.7, linewidth=0.5)
    ax1.axhline(E_EXPECTED, color='r', linestyle='--', linewidth=2, label=f'Expected: {E_EXPECTED:.2f}')
    ax1.axhline(best_energy, color='g', linestyle=':', linewidth=1, label=f'Best: {best_energy:.4f}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Error
    ax2 = axes[0, 1]
    errors = np.abs(np.array(energies) - E_EXPECTED) / E_EXPECTED * 100
    ax2.semilogy(epochs_arr, errors, 'g-', alpha=0.7, linewidth=0.5)
    ax2.axhline(1.0, color='r', linestyle='--', label='1% target')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Error (%)')
    ax2.set_title('Relative Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # E_loc std
    ax3 = axes[0, 2]
    ax3.semilogy(epochs_arr, energy_stds, 'm-', alpha=0.7, linewidth=0.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('E_loc std (log)')
    ax3.set_title('Local Energy Variance')
    ax3.grid(True, alpha=0.3)

    # Acceptance rate
    ax4 = axes[1, 0]
    ax4.plot(epochs_arr, acceptance_rates, 'c-', alpha=0.7, linewidth=0.5)
    ax4.axhline(0.30, color='r', linestyle='--', alpha=0.5)
    ax4.axhline(0.50, color='r', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Acceptance Rate')
    ax4.set_title('MCMC Acceptance Rate')
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.3)

    # Max V (escape monitoring)
    ax5 = axes[1, 1]
    ax5.semilogy(epochs_arr, max_V_history, 'orange', alpha=0.7, linewidth=0.5)
    ax5.axhline(30, color='r', linestyle='--', label='Reset threshold')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('Max V (log)')
    ax5.set_title('Maximum Potential (Escape Monitor)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # V distribution from fresh walkers
    ax6 = axes[1, 2]
    ax6.hist(all_V, bins=50, density=True, alpha=0.7, color='purple')
    ax6.axvline(10, color='r', linestyle='--', label='V=10 (well boundary)')
    ax6.set_xlabel('V(x)')
    ax6.set_ylabel('Density')
    ax6.set_title(f'Potential Distribution ({pct_in_well:.0f}% in well)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle(f'NNQS 6D Henon-Heiles: E = {final_E:.4f} ({final_error:.2f}% error)\n'
                 f'Best: {best_energy:.4f} ({best_error:.2f}%) | Training: {total_time/60:.1f} min',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('nnqs_6d_hh_convergence.png', dpi=150, bbox_inches='tight')
    print("Saved: nnqs_6d_hh_convergence.png")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    success = best_error < 1.0
    if success:
        print(f"\nSUCCESS: NNQS achieved {best_error:.2f}% error on 6D Henon-Heiles!")
    else:
        print(f"\nTarget not reached. Best error: {best_error:.2f}%")
        if best_error < 5.0:
            print("Within 5% - reasonable for this challenging anharmonic system.")

    print(f"\nKey observations:")
    print(f"  - Walkers stayed in well: {pct_in_well:.0f}% had V < 10")
    print(f"  - Fresh walker validation: E = {fresh_E_mean:.4f} ({fresh_error:.2f}% error)")
    print(f"  - E_loc variance: {fresh_E_std:.4f}")

    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
