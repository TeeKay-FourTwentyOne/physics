# Israel-Darmois Junction Conditions

## Overview

This document explains the junction conditions module implemented in `em_cylindrical/junction/`, which provides the Israel-Darmois formalism for matching spacetime regions across cylindrical hypersurfaces.

**References:**
- Israel, W. (1966). "Singular hypersurfaces and thin shells in general relativity." *Nuovo Cimento B* 44, 1-14.
- Darmois, G. (1927). *Mémorial des Sciences Mathématiques*, Fascicule 25.

---

## Physical Context

### The Matching Problem

In general relativity, we often need to "glue" together different spacetime regions:

- **Interior solutions** (ρ < R): May contain matter, currents, or specific field configurations
- **Exterior solutions** (ρ > R): Often vacuum or different field configurations

The junction conditions ensure this matching is physically consistent—that there are no gaps, overlaps, or unphysical discontinuities in the spacetime manifold.

### Applications

1. **Stellar models**: Match interior stellar solutions to exterior Schwarzschild/vacuum
2. **Cylindrical sources**: Match field-containing regions to vacuum exterior
3. **Thin shells**: Model shells of matter (domain walls, bubble walls)
4. **Gravitational wave sources**: Construct composite spacetimes for wave generation

---

## Mathematical Framework

### The Hypersurface Σ

For cylindrically symmetric spacetimes, we consider hypersurfaces at constant radial coordinate:

```
Σ: ρ = R (constant)
```

This is a 3-dimensional timelike surface with:
- **Tangent coordinates**: (Φ, z, t)
- **Normal direction**: ρ

### First Fundamental Form (Induced Metric)

The **induced metric** γ_ab describes the intrinsic geometry of Σ:

```
γ_ab = g_μν e^μ_a e^ν_b
```

For the Einstein-Rosen metric at ρ = R:

| Component | Expression | Physical Meaning |
|-----------|------------|------------------|
| γ_ΦΦ | -R² e^(-μ) | Angular distances on surface |
| γ_zz | -e^μ | Axial distances on surface |
| γ_tt | e^(λ-μ) | Time intervals on surface |

The signature (-,-,+) confirms this is a **timelike hypersurface**.

### Second Fundamental Form (Extrinsic Curvature)

The **extrinsic curvature** K_ab measures how Σ bends within the 4D spacetime:

```
K_ab = -½ ℒ_n γ_ab = -n_μ;ν e^μ_a e^ν_b
```

For the Einstein-Rosen metric:

```
K_ΦΦ = R e^((-μ-λ)/2) (1 - ½R μ_ρ)
K_zz = ½ e^((3μ-λ)/2) μ_ρ
K_tt = -½ e^((λ-μ)/2) (λ_ρ - μ_ρ)
```

Where μ_ρ = ∂μ/∂ρ and λ_ρ = ∂λ/∂ρ evaluated at ρ = R.

---

## The Junction Conditions

### First Condition (Darmois)

**The induced metric must be continuous:**

```
[γ_ab] = γ_ab⁺ - γ_ab⁻ = 0
```

This ensures:
- No gaps or overlaps in the spacetime manifold
- Observers on Σ measure the same distances/times from either side
- Coordinates can be consistently defined across the junction

**For Einstein-Rosen metric**, this requires:
- μ continuous at ρ = R
- λ continuous at ρ = R (or at least λ - μ continuous)

### Second Condition (Israel)

**The extrinsic curvature jump relates to surface stress-energy:**

```
[K_ab] = K_ab⁺ - K_ab⁻ = -8π(S_ab - ½γ_ab S)
```

Where:
- S_ab is the **surface stress-energy tensor** on Σ
- S = γ^ab S_ab is its trace

**Two cases:**

| Case | Condition | Physical Meaning |
|------|-----------|------------------|
| Smooth junction | [K_ab] = 0 | No matter on surface, smooth transition |
| Thin shell | [K_ab] ≠ 0 | Matter layer on surface (domain wall, shell) |

### Surface Stress-Energy

When [K_ab] ≠ 0, the surface stress-energy is:

```
S_ab = -(1/8π)([K_ab] - γ_ab [K])
```

Physical components:

| Quantity | Symbol | Formula | Meaning |
|----------|--------|---------|---------|
| Energy density | σ | S^t_t | Energy per unit area |
| Pressure (azimuthal) | p_Φ | -S^Φ_Φ | Tangential stress in Φ |
| Pressure (axial) | p_z | -S^z_z | Tangential stress in z |

### Energy Conditions

The module verifies physical plausibility:

| Condition | Requirement | Physical Meaning |
|-----------|-------------|------------------|
| Weak Energy (WEC) | σ ≥ 0 | Positive energy density |
| Dominant Energy (DEC) | σ ≥ \|p_i\| | Energy dominates pressure |

---

## Module Architecture

```
em_cylindrical/junction/
├── __init__.py              # Public API exports
├── hypersurface.py          # CylindricalHypersurface class
├── extrinsic_curvature.py   # ExtrinsicCurvature class
├── junction_conditions.py   # IsraelDarmoisConditions class
└── matched_solution.py      # MatchedSolution class
```

### Class Hierarchy

```
CylindricalHypersurface
    │
    ├── induced_metric()          → γ_ab (3×3)
    ├── induced_metric_inverse()  → γ^ab (3×3)
    ├── normal_vector()           → n^μ (4-vector)
    └── normal_covector()         → n_μ (4-vector)

ExtrinsicCurvature
    │
    ├── K_ab()                    → K_ab (3×3)
    ├── K_trace()                 → K = γ^ab K_ab
    ├── principal_curvatures()    → eigenvalues of K^a_b
    └── gaussian_curvature_contribution()

IsraelDarmoisConditions
    │
    ├── first_condition()         → [γ_ab] check
    ├── second_condition()        → [K_ab] check
    ├── surface_stress_energy()   → S_ab (3×3)
    ├── surface_energy_density()  → σ
    ├── surface_pressures()       → (p_Φ, p_z)
    └── verify_all()              → complete analysis

MatchedSolution
    │
    ├── mu(), lambda_(), psi()    → region-aware evaluation
    ├── junction()                → IsraelDarmoisConditions
    ├── is_valid()                → first condition check
    ├── has_thin_shell()          → thin shell detection
    └── shell_properties()        → thin shell analysis
```

---

## Usage Examples

### Basic Junction Condition Check

```python
from em_cylindrical import CaseISolution
from em_cylindrical.junction import IsraelDarmoisConditions

# Two solutions to match
sol_interior = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
sol_exterior = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

# Check junction conditions at R = 2.0
jc = IsraelDarmoisConditions(sol_interior, sol_exterior, radius=2.0)
result = jc.verify_all(t=1.0)

print(f"First condition satisfied: {result['first_condition']['satisfied']}")
print(f"Thin shell required: {result['second_condition']['thin_shell']}")
```

### Working with Matched Solutions

```python
from em_cylindrical import CaseISolution
from em_cylindrical.junction import MatchedSolution

# Create matched solution
sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
matched = MatchedSolution(sol, sol, junction_radius=2.0)

# Evaluate in either region automatically
mu_interior = matched.mu(1.5, 1.0)  # Uses interior solution
mu_exterior = matched.mu(2.5, 1.0)  # Uses exterior solution

# Check junction validity
print(f"Valid junction: {matched.is_valid(t=1.0)}")
print(f"Has thin shell: {matched.has_thin_shell(t=1.0)}")

# Get detailed junction analysis
print(matched.junction().summary(t=1.0))
```

### Analyzing Thin Shells

```python
from em_cylindrical import CaseISolution
from em_cylindrical.junction import IsraelDarmoisConditions

# Different solutions create discontinuity
sol_int = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
sol_ext = CaseISolution(alpha=1.5, beta=0.7, variant='t_only', m=1.5)

jc = IsraelDarmoisConditions(sol_int, sol_ext, radius=2.0)

# Get thin shell properties
sigma = jc.surface_energy_density(t=1.0)
p_Phi, p_z = jc.surface_pressures(t=1.0)

print(f"Surface energy density: σ = {sigma:.6f}")
print(f"Azimuthal pressure: p_Φ = {p_Phi:.6f}")
print(f"Axial pressure: p_z = {p_z:.6f}")

# Check energy conditions
print(f"Weak Energy Condition: {jc.weak_energy_condition(t=1.0)}")
print(f"Dominant Energy Condition: {jc.dominant_energy_condition(t=1.0)}")
```

### Hypersurface Geometry Analysis

```python
from em_cylindrical import CaseISolution
from em_cylindrical.junction import CylindricalHypersurface, ExtrinsicCurvature

sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
hs = CylindricalHypersurface(radius=2.0)

# Induced metric
gamma = hs.induced_metric(sol, t=1.0)
print(f"Induced metric diagonal: ({gamma[0,0]:.4f}, {gamma[1,1]:.4f}, {gamma[2,2]:.4f})")

# Extrinsic curvature
K = ExtrinsicCurvature(hs, sol)
K_ab = K.K_ab(t=1.0)
print(f"Extrinsic curvature diagonal: ({K_ab[0,0]:.4f}, {K_ab[1,1]:.4f}, {K_ab[2,2]:.4f})")
print(f"Mean curvature: K = {K.K_trace(t=1.0):.4f}")
```

---

## Physical Interpretation

### What the Junction Conditions Tell Us

1. **Metric Continuity** ([γ_ab] = 0):
   - The geometry of Σ is well-defined
   - Clocks and rulers work consistently across the junction
   - No "seams" visible to observers on the surface

2. **Extrinsic Curvature Jump** ([K_ab] ≠ 0):
   - The surface has "kink" when viewed from 4D
   - This kink requires matter support (thin shell)
   - The shell has energy density and pressures

### Cylindrical Thin Shells

For a cylindrical thin shell at ρ = R:

| Property | Physical Meaning |
|----------|------------------|
| σ > 0 | Positive mass shell (normal matter) |
| σ < 0 | Negative mass shell (exotic matter) |
| p_Φ > 0 | Tension in azimuthal direction |
| p_z > 0 | Tension in axial direction |

### Connection to Surface Currents

The junction module complements the existing `boundary.py` surface current calculations:

- **Surface currents** K^μ: Electromagnetic sources on Σ
- **Surface stress-energy** S_ab: Gravitational sources on Σ

Together, they fully characterize the physics of the boundary layer.

---

## Relation to Other Modules

### Compatibility with Wave Conversion

The junction conditions can be applied to wave conversion solutions:

```python
from em_cylindrical.wave_conversion import WaveConversion, LinearWaveSeed, ComplexLine
from em_cylindrical.junction import CylindricalHypersurface

# Create wave conversion solution
seed = LinearWaveSeed(amplitude=0.3, wavenumber=0.5)
wc = WaveConversion(seed=seed, submanifold=ComplexLine(theta=0.1))

# Analyze hypersurface geometry
# (requires wrapper to provide mu/lambda_ interface)
```

### Integration with Sourced Solutions

The `sources.py` module provides interior solutions with currents that can be matched to exterior vacuum using junction conditions.

---

## Mathematical Details

### Normal Vector Computation

For ρ = const surface, the unit normal covector is:

```
n_ρ = √|g_ρρ| = √(e^(λ-μ)) = e^((λ-μ)/2)
```

The contravariant normal (outward-pointing):

```
n^ρ = g^ρρ n_ρ = -e^(-(λ-μ)) · e^((λ-μ)/2) = -e^((μ-λ)/2)
```

We use the convention n^ρ > 0 for outward, so:

```
n^ρ = e^((μ-λ)/2)  (outward-pointing)
```

### Extrinsic Curvature Derivation

From K_ab = -½ ℒ_n γ_ab = -½ n^ρ ∂γ_ab/∂ρ:

**K_ΦΦ:**
```
∂γ_ΦΦ/∂ρ = ∂(-R²e^(-μ))/∂ρ = -2R e^(-μ) + R² e^(-μ) μ_ρ
K_ΦΦ = -½ e^((μ-λ)/2) (-2R e^(-μ) + R² e^(-μ) μ_ρ)
     = R e^((-μ-λ)/2) (1 - ½R μ_ρ)
```

**K_zz:**
```
∂γ_zz/∂ρ = ∂(-e^μ)/∂ρ = -e^μ μ_ρ
K_zz = -½ e^((μ-λ)/2) (-e^μ μ_ρ)
     = ½ e^((3μ-λ)/2) μ_ρ
```

**K_tt:**
```
∂γ_tt/∂ρ = ∂(e^(λ-μ))/∂ρ = e^(λ-μ) (λ_ρ - μ_ρ)
K_tt = -½ e^((μ-λ)/2) e^(λ-μ) (λ_ρ - μ_ρ)
     = -½ e^((λ-μ)/2) (λ_ρ - μ_ρ)
```

---

## Testing

Run the junction condition tests:

```bash
python -m pytest tests/test_junction.py -v
```

Run the demo:

```bash
python examples/junction_demo.py
```

---

## References

1. Israel, W. (1966). "Singular hypersurfaces and thin shells in general relativity." *Nuovo Cimento B* 44, 1-14.

2. Darmois, G. (1927). "Les équations de la gravitation einsteinienne." *Mémorial des Sciences Mathématiques*, Fascicule 25.

3. Poisson, E. (2004). *A Relativist's Toolkit: The Mathematics of Black-Hole Mechanics*. Cambridge University Press. Chapter 3.

4. Mars, M. & Senovilla, J.M.M. (1993). "Geometry of general hypersurfaces in spacetime: junction conditions." *Classical and Quantum Gravity* 10, 1865.

5. Misra, M. & Radhakrishna, L. (1962). "Some Electromagnetic Fields of Cylindrical Symmetry." *Proc. Nat. Inst. Sci. India* 28A(4), 632-645.
