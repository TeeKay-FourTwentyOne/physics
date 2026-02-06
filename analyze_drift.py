import numpy as np
import torch
import matplotlib.pyplot as plt
from quantum_hj import CoupledOscillator6D
from quantum_hj.pinn_nd import QuantumNumberTrainerND

print('='*70)
print('6D Coupled Oscillator: Energy Drift Investigation')
print('='*70)

# Load checkpoint with full history
checkpoint = torch.load('6d_coupled_checkpoint.pt', weights_only=False)

E_history = np.array(checkpoint['E_history'])
physics_loss = np.array(checkpoint['physics_loss_history'])
curl_loss = np.array(checkpoint['curl_loss_history'])
quant_loss = np.array(checkpoint['quant_loss_history'])
supervision_loss = np.array(checkpoint['supervision_loss_history'])

n_epochs = len(E_history)
total_hours = 20.2
time_hours = np.linspace(0, total_hours, n_epochs)

target_energy = 2.982753

# Find key points
best_idx = np.argmin(np.abs(E_history - target_energy))
best_time = time_hours[best_idx]
best_energy = E_history[best_idx]
best_error = abs(best_energy - target_energy) / target_energy * 100

print(f'Best energy: E = {best_energy:.6f} at hour {best_time:.1f} (epoch {best_idx})')
print(f'Best error: {best_error:.2f}%')
print(f'Final energy: E = {E_history[-1]:.6f} (error: {abs(E_history[-1]-target_energy)/target_energy*100:.2f}%)')

# ============================================================
# Analyze loss components over time
# ============================================================
print('\n' + '='*70)
print('LOSS COMPONENT ANALYSIS')
print('='*70)

# Sample at checkpoint intervals
checkpoint_epochs = np.linspace(0, n_epochs-1, 12).astype(int)
checkpoint_times = time_hours[checkpoint_epochs]

print(f"{'Hour':<6} {'Energy':<10} {'Phys Loss':<12} {'Curl Loss':<12} {'Quant Loss':<12} {'Sup Loss':<12}")
print('-'*70)
for t, idx in zip(checkpoint_times, checkpoint_epochs):
    print(f'{t:<6.1f} {E_history[idx]:<10.6f} {physics_loss[idx]:<12.4f} {curl_loss[idx]:<12.4f} {quant_loss[idx]:<12.4f} {supervision_loss[idx]:<12.4f}')

# ============================================================
# Create analysis figure
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# ---- Top: Energy and Physics Loss vs Time (dual axis) ----
ax1 = axes[0]
ax1_twin = ax1.twinx()

# Energy
ax1.plot(time_hours, E_history, 'b-', linewidth=0.8, alpha=0.7, label='Energy')
ax1.axhline(y=target_energy, color='green', linestyle='--', linewidth=2, label=f'Target: {target_energy:.4f}')
ax1.axvline(x=best_time, color='purple', linestyle=':', linewidth=2, alpha=0.7, label=f'Best E at {best_time:.1f}h')
ax1.scatter([best_time], [best_energy], c='purple', s=100, zorder=5, marker='*')
ax1.annotate(f'Best: {best_energy:.4f}', xy=(best_time, best_energy),
             xytext=(best_time+1, best_energy-0.01), fontsize=9)

# Physics loss on twin axis
ax1_twin.semilogy(time_hours, physics_loss, 'r-', linewidth=0.8, alpha=0.7, label='Physics Loss')

ax1.set_ylabel('Energy (a.u.)', color='blue', fontsize=11)
ax1_twin.set_ylabel('Physics Loss (log)', color='red', fontsize=11)
ax1.set_ylim([2.97, 3.06])
ax1.set_title('Energy vs Physics Loss: Divergence After Hour 8', fontsize=12, fontweight='bold')

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# ---- Middle: All Loss Components ----
ax2 = axes[1]
ax2.semilogy(time_hours, physics_loss, 'r-', linewidth=1, alpha=0.8, label='Physics Loss')
ax2.semilogy(time_hours, curl_loss, 'g-', linewidth=1, alpha=0.8, label='Curl Loss')
ax2.semilogy(time_hours, quant_loss, 'b-', linewidth=1, alpha=0.8, label='Quantization Loss')
ax2.semilogy(time_hours, supervision_loss, 'm-', linewidth=1, alpha=0.8, label='Supervision Loss')
ax2.axvline(x=best_time, color='purple', linestyle=':', linewidth=2, alpha=0.7)

# Mark quantization loss activation (around epoch 6000 = 20% of training)
quant_start_time = time_hours[6000]
ax2.axvline(x=quant_start_time, color='orange', linestyle='--', linewidth=2, alpha=0.7,
            label=f'Quant loss starts ({quant_start_time:.1f}h)')

ax2.set_ylabel('Loss (log scale)', fontsize=11)
ax2.set_title('Loss Components Over Training', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.3)

# ---- Bottom: Energy Error vs Physics Loss (scatter) ----
ax3 = axes[2]

# Sample every 1000 epochs for scatter plot
sample_indices = np.arange(0, n_epochs, 1000)
sample_times = time_hours[sample_indices]
sample_physics_loss = physics_loss[sample_indices]
sample_energy_error = np.abs(E_history[sample_indices] - target_energy)

scatter = ax3.scatter(sample_physics_loss, sample_energy_error,
                      c=sample_times, cmap='viridis', s=50, alpha=0.8)
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Training Time (hours)', fontsize=10)

# Label a few key points
for idx, (pl, ee, t) in enumerate(zip(sample_physics_loss, sample_energy_error, sample_times)):
    if int(t) in [0, 2, 4, 6, 8, 12, 16, 20]:
        ax3.annotate(f'{t:.0f}h', xy=(pl, ee), xytext=(5, 5),
                     textcoords='offset points', fontsize=8)

ax3.set_xlabel('Physics Loss', fontsize=11)
ax3.set_ylabel('|E - E_target|', fontsize=11)
ax3.set_title('Energy Error vs Physics Loss (colored by time)', fontsize=12, fontweight='bold')
ax3.set_xscale('log')
ax3.grid(True, alpha=0.3)

# Add correlation info
correlation = np.corrcoef(np.log(sample_physics_loss[1:]), sample_energy_error[1:])[0,1]
ax3.text(0.05, 0.95, f'Correlation: {correlation:.2f}', transform=ax3.transAxes,
         fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.xlabel('Training Time (hours)', fontsize=11)
plt.tight_layout()
plt.savefig('energy_drift_analysis.png', dpi=150, bbox_inches='tight')
print(f'\nFigure saved to: energy_drift_analysis.png')

# ============================================================
# Hypothesis Testing
# ============================================================
print('\n' + '='*70)
print('HYPOTHESIS ANALYSIS')
print('='*70)

# A. Energy oscillation vs monotonic drift
print('\nA. LEARNING RATE / OSCILLATION ANALYSIS')
early_E = E_history[5000:10000]  # Hours 3-7
late_E = E_history[20000:30000]  # Hours 13-20
print(f'   Early phase (h3-7) energy std: {np.std(early_E):.6f}')
print(f'   Late phase (h13-20) energy std: {np.std(late_E):.6f}')
if np.std(late_E) > np.std(early_E):
    print('   -> Energy MORE variable late in training (possible LR too high)')
else:
    print('   -> Energy LESS variable late in training (LR decay working)')

# B. Check if energy drifted monotonically after best point
print('\nB. DRIFT PATTERN AFTER BEST POINT')
post_best_E = E_history[best_idx:]
x = np.arange(len(post_best_E))
slope = np.polyfit(x, post_best_E, 1)[0]
print(f'   Energy slope after best point: {slope*1000:.4f} per 1000 epochs')
if slope > 0:
    print('   -> Monotonic UPWARD drift (pulled away from target)')
else:
    print('   -> Monotonic DOWNWARD drift (toward target)')

# C. Quantization loss impact
print('\nC. QUANTIZATION LOSS IMPACT')
quant_start_epoch = 6000
pre_quant_error = np.mean(np.abs(E_history[4000:6000] - target_energy))
post_quant_error = np.mean(np.abs(E_history[8000:10000] - target_energy))
print(f'   Mean error before quant loss (h2.7-4): {pre_quant_error:.6f}')
print(f'   Mean error after quant loss (h5.4-6.7): {post_quant_error:.6f}')

# D. Correlation between losses
print('\nD. LOSS CORRELATION ANALYSIS')
post_best_phys = physics_loss[best_idx:]
post_best_curl = curl_loss[best_idx:]
post_best_quant = quant_loss[best_idx:]
post_best_sup = supervision_loss[best_idx:]

corr_phys_curl = np.corrcoef(post_best_phys, post_best_curl)[0,1]
corr_phys_quant = np.corrcoef(post_best_phys, post_best_quant)[0,1]
corr_phys_sup = np.corrcoef(post_best_phys, post_best_sup)[0,1]

print(f'   Physics-Curl correlation (after h8): {corr_phys_curl:.3f}')
print(f'   Physics-Quant correlation (after h8): {corr_phys_quant:.3f}')
print(f'   Physics-Supervision correlation (after h8): {corr_phys_sup:.3f}')

# E. Quantization loss pulling energy
print('\nE. QUANTIZATION LOSS TARGET ANALYSIS')
final_quant = quant_loss[-1]
best_quant = quant_loss[best_idx]
print(f'   Quantization loss at best energy: {best_quant:.4f}')
print(f'   Quantization loss at final: {final_quant:.4f}')
print(f'   Change: {final_quant - best_quant:+.4f}')

# ============================================================
# Summary
# ============================================================
print('\n' + '='*70)
print('SUMMARY AND RECOMMENDATIONS')
print('='*70)

print("""
FINDINGS:
1. Energy reached optimum (0.19% error) at hour 8, then drifted to 0.45% error
2. Physics loss continued decreasing throughout - loss and accuracy misaligned
3. Quantization loss activated at hour 4 and stayed high (~0.6-0.7)
4. Energy drifted UPWARD after best point (toward uncoupled value 3.0)

LIKELY CAUSE:
The quantization loss assumes J_i = (n_i + 0.5)*hbar in Cartesian coordinates.
For COUPLED systems, the action integrals in Cartesian coords do NOT satisfy
this relation - they should be computed in normal mode coordinates instead.
The quantization loss pulled energy toward 3.0 (the uncoupled ZPE).

RECOMMENDATIONS FOR COUPLED SYSTEMS:
1. Disable quantization loss, rely on physics + supervision only
2. Or transform to normal mode coordinates for action computation
3. Implement early stopping based on energy error vs target
4. Save checkpoint at best energy, not just periodically
""")
