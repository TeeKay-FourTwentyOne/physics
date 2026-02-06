"""Quick check of 12D individual trial results."""
import torch
import numpy as np

data = torch.load('scaling_12d.pt', weights_only=False)

print("=" * 70)
print("12D INDIVIDUAL TRIAL RESULTS")
print("=" * 70)

print(f"\n| Trial | Seed | Training E | Fresh E    | Fresh Std  | Diff (%) |")
print(f"|-------|------|------------|------------|------------|----------|")

for i, r in enumerate(data['individual_results']):
    train_E = r['final_energy']
    fresh_E = r['fresh_energy']
    fresh_std = r['fresh_std']
    diff_pct = abs(fresh_E - train_E) / train_E * 100

    flag = "***" if fresh_std > 1.0 else ""
    print(f"|   {i+1}   | {r['seed']:4d} | {train_E:10.4f} | {fresh_E:10.4f} | {fresh_std:10.4f} | {diff_pct:8.2f} | {flag}")

print("\nAnalysis:")
for i, r in enumerate(data['individual_results']):
    if r['fresh_std'] > 10.0:
        print(f"  Trial {i+1}: CATASTROPHIC fresh_std = {r['fresh_std']:.2f}")
    elif r['fresh_std'] > 1.0:
        print(f"  Trial {i+1}: HUGE fresh_std = {r['fresh_std']:.2f} - sampling problem")
    elif r['fresh_std'] > 0.5:
        print(f"  Trial {i+1}: High fresh_std = {r['fresh_std']:.2f}")
    else:
        print(f"  Trial {i+1}: Normal fresh_std = {r['fresh_std']:.2f}")

# Compare with 24D
print("\n" + "=" * 70)
print("24D INDIVIDUAL TRIAL RESULTS (for comparison)")
print("=" * 70)

data24 = torch.load('scaling_24d.pt', weights_only=False)

print(f"\n| Trial | Seed | Training E | Fresh E    | Fresh Std  | Diff (%) |")
print(f"|-------|------|------------|------------|------------|----------|")

for i, r in enumerate(data24['individual_results']):
    train_E = r['final_energy']
    fresh_E = r['fresh_energy']
    fresh_std = r['fresh_std']
    diff_pct = abs(fresh_E - train_E) / train_E * 100
    print(f"|   {i+1}   | {r['seed']:4d} | {train_E:10.4f} | {fresh_E:10.4f} | {fresh_std:10.4f} | {diff_pct:8.2f} |")

print("\n" + "=" * 70)
print("HYPERPARAMETER COMPARISON")
print("=" * 70)
print("""
| Parameter      | 12D (original)     | 24D (worked)       |
|----------------|--------------------|--------------------|
| hidden_size    | 896 (512+64*6)     | 512 (capped)       |
| n_walkers      | 10,000             | 5,000              |
| Network params | ~3.2M              | ~1.1M              |

12D used a LARGER network with MORE data, but worse results.
24D used a SMALLER network with LESS data, but better results.

Hypothesis: 12D network (896 hidden) may be overparameterized for the
training budget, leading to poor generalization despite good training loss.
""")
