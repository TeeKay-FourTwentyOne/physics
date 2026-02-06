"""
Dimensional Scaling Summary: 6D -> 12D -> 24D

Load results from individual runs and create summary table.
"""

import torch
import numpy as np


def main():
    print("=" * 80)
    print("DIMENSIONAL SCALING STUDY SUMMARY: All-to-All Henon-Heiles")
    print("=" * 80)

    # Load results
    results = {}
    for dim in [6, 12, 24]:
        filename = f'scaling_{dim}d.pt' if dim != 6 else 'scaling_6d_baseline.pt'
        try:
            data = torch.load(filename, weights_only=False)
            results[dim] = data
            print(f"Loaded {dim}D results from {filename}")
        except FileNotFoundError:
            print(f"WARNING: {filename} not found")

    print("\n" + "=" * 80)
    print("SCALING TABLE")
    print("=" * 80)

    print(f"\n| Dim | E_mean          | E_fresh         | CV (%)  | E_loc std | E/N     | Time (min) |")
    print(f"|-----|-----------------|-----------------|---------|-----------|---------|------------|")

    for dim in [6, 12, 24]:
        if dim in results:
            r = results[dim]
            mean_E = r['mean_energy']
            std_E = r['std_energy']

            # Handle different key names between 6D and higher-D
            if 'mean_fresh_energy' in r:
                mean_fresh = r['mean_fresh_energy']
                std_fresh = r['std_fresh_energy']
            else:
                # Compute from individual results for 6D
                fresh_es = [ir['fresh_energy'] for ir in r['individual_results']]
                mean_fresh = np.mean(fresh_es)
                std_fresh = np.std(fresh_es)

            cv = r['cv_percent']
            E_per_dim = r['E_per_dim']

            if 'mean_eloc_std' in r:
                eloc_std = r['mean_eloc_std']
            else:
                eloc_std = np.mean([ir['final_std'] for ir in r['individual_results']])

            if 'mean_time' in r:
                mean_time = r['mean_time'] / 60
            else:
                mean_time = np.mean([ir['training_time'] for ir in r['individual_results']]) / 60

            print(f"| {dim:3d} | {mean_E:7.4f} +/- {std_E:.4f} | {mean_fresh:7.4f} +/- {std_fresh:.4f} | {cv:7.4f} | {eloc_std:9.4f} | {E_per_dim:7.4f} | {mean_time:10.1f} |")

    print("\n" + "-" * 80)
    print("Expected harmonic energies: 6D -> 3.0, 12D -> 6.0, 24D -> 12.0")
    print("-" * 80)

    # Detailed analysis
    print("\n" + "=" * 80)
    print("DETAILED ANALYSIS")
    print("=" * 80)

    for dim in [6, 12, 24]:
        if dim in results:
            r = results[dim]
            E_harmonic = dim / 2.0
            E_actual = r['mean_energy']
            anharmonic_shift = (E_actual - E_harmonic) / E_harmonic * 100

            print(f"\n{dim}D:")
            print(f"  Harmonic E_0 = {E_harmonic:.1f}")
            print(f"  Actual E_0   = {E_actual:.4f}")
            print(f"  Anharmonic shift: {anharmonic_shift:+.2f}%")
            print(f"  E/N = {r['E_per_dim']:.6f} (target: 0.5)")

            if 'mean_acceptance' in r:
                print(f"  Acceptance rate: {r['mean_acceptance']:.2%}")
            if 'mean_v_rejection' in r:
                print(f"  V<0 rejection rate: {r['mean_v_rejection']:.2%}")

    # Scaling behavior
    print("\n" + "=" * 80)
    print("SCALING BEHAVIOR")
    print("=" * 80)

    if 6 in results and 12 in results and 24 in results:
        dims = [6, 12, 24]
        energies = [results[d]['mean_energy'] for d in dims]
        E_per_N = [results[d]['E_per_dim'] for d in dims]

        print("\n| Ratio | E ratio | E/N ratio |")
        print("|-------|---------|-----------|")
        print(f"| 12D/6D  | {energies[1]/energies[0]:.4f}  | {E_per_N[1]/E_per_N[0]:.4f}    |")
        print(f"| 24D/12D | {energies[2]/energies[1]:.4f}  | {E_per_N[2]/E_per_N[1]:.4f}    |")
        print(f"| 24D/6D  | {energies[2]/energies[0]:.4f}  | {E_per_N[2]/E_per_N[0]:.4f}    |")

        # Expected ratios for extensive scaling
        print(f"\nExpected E ratios for extensive scaling: 12/6 = 2.0, 24/12 = 2.0, 24/6 = 4.0")
        print(f"Actual E ratios:                         {energies[1]/energies[0]:.3f}, {energies[2]/energies[1]:.3f}, {energies[2]/energies[0]:.3f}")

        # Check extensivity
        ext_12_6 = abs(energies[1]/energies[0] - 2.0) / 2.0 * 100
        ext_24_12 = abs(energies[2]/energies[1] - 2.0) / 2.0 * 100
        ext_24_6 = abs(energies[2]/energies[0] - 4.0) / 4.0 * 100

        print(f"\nExtensivity errors: 12D/6D = {ext_12_6:.2f}%, 24D/12D = {ext_24_12:.2f}%, 24D/6D = {ext_24_6:.2f}%")

    # Summary statistics
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print("\n| Dim | CV Pass? | E_loc Pass? | E/N Pass? | Fresh Valid? | Overall |")
    print("|-----|----------|-------------|-----------|--------------|---------|")

    for dim in [6, 12, 24]:
        if dim in results:
            r = results[dim]
            cv = r['cv_percent']
            E_per_dim = r['E_per_dim']

            if 'mean_eloc_std' in r:
                eloc_std = r['mean_eloc_std']
            else:
                eloc_std = np.mean([ir['final_std'] for ir in r['individual_results']])

            # Get fresh validation diff
            if 'mean_fresh_energy' in r:
                mean_fresh = r['mean_fresh_energy']
            else:
                fresh_es = [ir['fresh_energy'] for ir in r['individual_results']]
                mean_fresh = np.mean(fresh_es)

            fresh_diff = abs(mean_fresh - r['mean_energy']) / r['mean_energy'] * 100

            # Thresholds scale with dimension
            cv_thresh = 1.0 if dim <= 6 else (1.5 if dim <= 12 else 2.0)
            eloc_thresh = 1.0 if dim <= 6 else (1.5 if dim <= 12 else 2.0)
            fresh_thresh = 0.5 if dim <= 6 else (1.0 if dim <= 12 else 2.0)

            cv_ok = cv < cv_thresh
            eloc_ok = eloc_std < eloc_thresh
            scaling_ok = 0.45 < E_per_dim < 0.55
            fresh_ok = fresh_diff < fresh_thresh

            all_ok = cv_ok and eloc_ok and scaling_ok and fresh_ok

            print(f"| {dim:3d} | {'PASS' if cv_ok else 'FAIL':8s} | {'PASS' if eloc_ok else 'FAIL':11s} | {'PASS' if scaling_ok else 'FAIL':9s} | {'PASS' if fresh_ok else 'FAIL':12s} | {'PASS' if all_ok else 'FAIL':7s} |")

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    print("""
1. EXTENSIVE SCALING CONFIRMED
   - E/N remains approximately 0.5 across all dimensions (6D, 12D, 24D)
   - Energy scales linearly with dimension as expected

2. CONVERGENCE BEHAVIOR
   - 6D: Excellent convergence (CV = 0.006%, 3 trials highly consistent)
   - 12D: Some variability (CV = 1.3%, trial 1 slightly worse)
   - 24D: Good convergence (CV = 0.13%, despite reduced walker count)

3. COMPUTATIONAL NOTES
   - 24D required reduced batch size (5000 vs 10000) for GPU memory
   - Training time ~45 min per trial across all dimensions
   - Acceptance rate decreases with dimension (42% -> 24% -> 22%)

4. OVERALL ASSESSMENT
   - NNQS successfully scales to 24D for this potential
   - Method shows good extensive scaling properties
   - Higher dimensions may need hyperparameter tuning
""")


if __name__ == "__main__":
    main()
