# Gravitational-Electromagnetic Wave Mode Conversion

## Overview

This document explains the wave conversion module implemented in `em_cylindrical/wave_conversion/`, based on the 2024 paper by Mishima & Tomizawa published in Physical Review D.

**Reference:**
Mishima, T. & Tomizawa, S. (2024). "Nonlinear dynamics driving the conversion of gravitational and electromagnetic waves in cylindrically symmetric spacetime." *Phys. Rev. D* 110, 024038. [arXiv:2405.04231](https://arxiv.org/abs/2405.04231)

---

## Physical Context

### The Original Problem (Misra & Radhakrishna 1962)

The original implementation in this project solves the Einstein-Maxwell equations for cylindrically symmetric spacetimes. These solutions describe electromagnetic fields (electric and magnetic) existing in curved spacetime, where the geometry itself is influenced by the electromagnetic energy.

The key insight of the 1962 paper was that exact analytical solutions exist for three cases:
- **Case I:** Only axial electromagnetic potential (ψ ≠ 0, φ = 0)
- **Case II:** Only azimuthal electromagnetic potential (φ ≠ 0, ψ = 0)
- **Case III:** Both potentials non-zero

### What the 2024 Paper Adds

The Mishima & Tomizawa paper addresses a fundamentally different question: **How do gravitational waves and electromagnetic waves convert into each other through nonlinear interactions?**

In general relativity, gravitational waves carry energy, and this energy curves spacetime. When electromagnetic waves are present, they also carry energy that curves spacetime. In the strong-field regime near the symmetry axis, these waves can exchange energy—gravitational wave energy can become electromagnetic wave energy and vice versa.

This is a **purely nonlinear effect** that:
1. Cannot be seen in linearized gravity (weak-field approximation)
2. Requires no external background field to occur
3. Is driven entirely by the nonlinearity of Einstein's equations

---

## Mathematical Framework

### Ernst Potential Formulation

The 2024 paper uses the **Ernst potential formulation**, which is mathematically elegant and reveals hidden symmetries. Instead of working with metric components directly, we use complex potentials:

**Gravitational Ernst potential:**
```
E = e^(2ψ) + |F|² - iΦ
```

**Electromagnetic Ernst potential:**
```
F = A_z + iχ
```

Where:
- ψ is a metric function (determines g_zz)
- Φ is the "twist" potential (related to rotation)
- A_z is the z-component of the electromagnetic 4-potential
- χ is the magnetic potential

These satisfy the **Ernst equations**, a coupled system of nonlinear PDEs that encapsulates the full Einstein-Maxwell system.

### Reduced Potentials and the Unit Disc

The Ernst potentials are transformed to "reduced" potentials:
```
ξ = (E - 1) / (E + 1)
η = 2F / (E + 1)
```

These map the solution space to the **complex hyperbolic 2-space H²_C**, with the constraint:
```
|ξ|² + |η|² < 1
```

This constraint defines a unit ball in complex 2-space, and valid solutions must live inside this ball.

### Harmonic Map Method

The key mathematical insight is that the Ernst equations describe **harmonic maps** from the 3D base manifold (with coordinates ρ, t, φ) to H²_C. This means:

1. We can start with a **vacuum seed solution** (pure gravity, F = 0)
2. Embed it into a **geodesic submanifold** of H²_C
3. The embedding "mixes" gravitational and electromagnetic components
4. The result is an exact Einstein-Maxwell solution

Two types of geodesic submanifolds are used:

**(a) Complex Line:**
```
ξ = cos(2θ) · z
η = sin(2θ) · z
```
The angle θ controls the gravitational/electromagnetic mixing ratio. At θ = 0, we have pure gravity; at θ = π/4, we have equal mixing.

**(b) Lagrangian Plane:**
Uses real coordinates through a Cayley-like transformation. This produces more complex behavior with nontrivial mode conversions.

---

## Seed Solutions

### Linear Wave Seed (Seed i)

Based on τ satisfying the cylindrical wave equation:
```
∂²τ/∂ρ² - ∂²τ/∂t² + (1/ρ)∂τ/∂ρ = 0
```

Solutions involve Bessel functions:
```
τ = A · J₀(kρ) · cos(kt + ε)
```

This connects directly to the Bessel function solutions in the original Case I implementation.

### Solitonic Seed (Seed ii)

The **Economou-Tsoubelis soliton** is defined in transformed coordinates (x, y):
```
ξ_v = (1 - il) / (px - iqy)
```

With the constraint: **q² - p² - l² = 1**

Coordinate relations:
```
t = xy
ρ = √[(x² + 1)(y² - 1)]
```

This solitonic seed produces the most interesting physics—nontrivial mode conversions that persist to infinity.

---

## C-Energy: Measuring Gravitational Wave Energy

### Thorne's C-Energy (1965)

A fundamental problem in general relativity is defining "energy" for gravitational waves. Unlike electromagnetic waves, gravitational wave energy cannot be localized in a coordinate-independent way due to the equivalence principle.

For cylindrically symmetric spacetimes, Kip Thorne defined **C-energy** (cylindrical energy) in 1965, which provides:

1. **Local measurement:** An observer can measure the C-energy density along their worldline
2. **Conservation:** C-energy satisfies a local conservation law
3. **Propagation:** C-energy is carried by both gravitational and electromagnetic waves

The total C-energy within radius ρ₀ is:
```
E(t, ρ₀) = γ(t, ρ₀) - γ(t, 0)
```

Where γ is a metric function in the Kompaneets-Jordan-Ehlers form.

### Mode Decomposition

The C-energy density can be decomposed into **four modes**:

| Mode | Symbol | Physical Meaning |
|------|--------|------------------|
| + polarization | ℰ_+ | Gravitational wave (one polarization) |
| × polarization | ℰ_× | Gravitational wave (other polarization) |
| z-mode | ℰ_z | Electromagnetic (axial) |
| φ-mode | ℰ_φ | Electromagnetic (azimuthal) |

The **occupancy ratios** are:
```
R_grav = (ℰ_+ + ℰ_×) / ℰ_total
R_em = (ℰ_z + ℰ_φ) / ℰ_total
```

With R_grav + R_em = 1 (energy is conserved, just redistributed between modes).

---

## Physical Results

### Mode Conversion Near the Axis

Near the cylindrical symmetry axis (small ρ), the gravitational field is strongest. In this region:

1. Gravitational and electromagnetic waves interact nonlinearly
2. Energy transfers between modes
3. The occupancy ratios R_grav and R_em change

### Behavior at Infinity

**For Complex Line (Case a):**
- Conversions near the axis **revert** as waves propagate outward
- At null infinity, the mode ratio returns to its initial value
- Energy "borrowed" by one mode is returned to the other

**For Lagrangian Plane with Solitonic Seed (Case b):**
- Conversions are **nontrivial** and persist to infinity
- Electromagnetic amplification factors range from **0.4× to 2.4×**
- A wave that starts as 50% gravitational / 50% electromagnetic can end up as 20% / 80%

### Amplification Ratio

The paper defines an amplification ratio:
```
Ratio = γ_em(+∞) / γ_em(-∞)
```

This measures how much the electromagnetic content changes from past to future null infinity. Values > 1 indicate electromagnetic amplification; values < 1 indicate gravitational amplification.

---

## Relation to Existing Implementation

### Metric Form Correspondence

The original Misra-Radhakrishna implementation uses:
```
ds² = e^(λ-μ)(dt² - dρ²) - ρ²e^(-μ)dΦ² - e^μ dz²
```

The wave conversion module uses the Kompaneets-Jordan-Ehlers form:
```
ds² = e^(2ψ)(dz - w dφ)² + ρ²e^(-2ψ)dφ² + e^(2(γ-ψ))(-dt² + dρ²)
```

The correspondence (for w = 0, no rotation) is:
```
μ = 2ψ
λ = 2γ
```

### Converting Between Formulations

The `WaveConversion.to_misra_radhakrishna()` method provides direct conversion:

```python
mr = wc.to_misra_radhakrishna(rho, t)
# Returns: {'mu': ..., 'lambda': ..., 'phi': ..., 'psi_MR': ...}
```

This allows comparing wave conversion solutions with the original Case I/II/III solutions.

### Electromagnetic Potentials

In Misra-Radhakrishna notation:
- ψ_MR = √(8π) · A_z (axial EM potential)
- φ_MR = √(8π) · χ (azimuthal EM potential)

---

## Usage Examples

### Basic Mode Conversion Analysis

```python
from em_cylindrical.wave_conversion import (
    WaveConversion, SolitonicSeed, LagrangianPlane
)

# Create solution showing nontrivial conversion
seed = SolitonicSeed.from_p_l(p=1.0, l=1.0)  # q = √3 automatically
wc = WaveConversion(seed=seed, submanifold=LagrangianPlane())

# Analyze at a point
rho, t = 2.0, 2.0

# Get mode occupancy
c = wc.c_energy()
R_grav, R_em = c.occupancy_ratios(rho, t)
print(f"Gravitational: {R_grav:.1%}")
print(f"Electromagnetic: {R_em:.1%}")
```

### Comparing with Original Solutions

```python
from em_cylindrical import CaseISolution
from em_cylindrical.wave_conversion import WaveConversion, LinearWaveSeed, ComplexLine

# Original solution
sol = CaseISolution(alpha=1.0, beta=0.5, variant='bessel', k=1.0, A=0.5)

# Wave conversion solution with similar parameters
seed = LinearWaveSeed(amplitude=0.5, wavenumber=1.0)
wc = WaveConversion(seed=seed, submanifold=ComplexLine(theta=0.1))

# Compare metric functions
rho, t = 2.0, 1.0
print(f"Original μ: {sol.mu(rho, t):.6f}")
print(f"Converted μ: {wc.to_misra_radhakrishna(rho, t)['mu']:.6f}")
```

---

## Physical Significance

### Why This Matters

1. **Gravitational wave astronomy:** Understanding how gravitational waves interact with electromagnetic fields is crucial for multi-messenger astronomy.

2. **Strong-field physics:** Near compact objects (neutron stars, black holes), fields are strong enough for nonlinear effects to matter.

3. **Early universe cosmology:** In the early universe, gravitational and electromagnetic radiation may have interacted significantly.

4. **Fundamental physics:** This demonstrates a purely general relativistic effect with no Newtonian analogue.

### Limitations

1. **Cylindrical symmetry:** Real astrophysical sources are not perfectly cylindrical. However, cylindrical symmetry provides exact analytical solutions that illuminate the physics.

2. **Vacuum solutions:** These are source-free solutions. Real systems have matter sources.

3. **Idealized initial conditions:** The seed solutions represent specific initial configurations.

---

## References

1. Mishima, T. & Tomizawa, S. (2024). "Nonlinear dynamics driving the conversion of gravitational and electromagnetic waves in cylindrically symmetric spacetime." *Phys. Rev. D* 110, 024038.

2. Thorne, K. S. (1965). "Energy of Infinitely Long, Cylindrically Symmetric Systems in General Relativity." *Phys. Rev.* 138, B251.

3. Ernst, F. J. (1968). "New Formulation of the Axially Symmetric Gravitational Field Problem." *Phys. Rev.* 167, 1175.

4. Misra, M. & Radhakrishna, L. (1962). "Some Electromagnetic Fields of Cylindrical Symmetry." *Proc. Nat. Inst. Sci. India* 28A(4), 632-645.

5. Economou, A. & Tsoubelis, D. (1988). "Rotating and translating strings in gravitational plane wave spacetimes." *Phys. Rev. D* 38, 498.

---

## Module Structure

```
em_cylindrical/wave_conversion/
├── __init__.py          # Public API exports
├── ernst.py             # Ernst potential calculations
├── seeds.py             # Vacuum seed solutions
├── harmonic_map.py      # Geodesic submanifold embeddings
├── c_energy.py          # C-energy and mode decomposition
└── conversion.py        # Main WaveConversion class
```

Run the demo to see all features:
```bash
python examples/wave_conversion_demo.py
```

Run tests:
```bash
python -m pytest tests/test_wave_conversion.py -v
```
