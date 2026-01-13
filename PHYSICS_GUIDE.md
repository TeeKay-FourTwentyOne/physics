# Physics Guide: Einstein-Maxwell Cylindrical Symmetry

This document explains the physical meaning of the quantities computed by this library.

## Overview

This library implements exact solutions of the **Einstein-Maxwell equations** for electromagnetic fields with **cylindrical symmetry** in **curved spacetime** (general relativity).

The solutions come from the 1962 paper by Misra & Radhakrishna.

---

## Coordinate System

**Coordinates**: (ρ, Φ, z, t)

| Coordinate | Meaning | Range |
|------------|---------|-------|
| ρ (rho) | Radial distance from axis | 0 < ρ < ∞ |
| Φ (Phi) | Azimuthal angle | 0 ≤ Φ < 2π |
| z | Axial coordinate | -∞ < z < ∞ |
| t | Time | -∞ < t < ∞ |

**Index convention**: (0, 1, 2, 3) = (ρ, Φ, z, t)

---

## The Metric (Spacetime Geometry)

The **Einstein-Rosen metric** describes the geometry of spacetime:

```
ds² = e^(λ-μ)(dt² - dρ²) - ρ²e^(-μ)dΦ² - e^μdz²
```

### Metric Functions

| Function | Symbol | Meaning |
|----------|--------|---------|
| `mu(ρ, t)` | μ | Controls z-direction geometry and field coupling |
| `lambda_(ρ, t)` | λ | Controls overall spacetime curvature |

### Metric Tensor g_μν

The metric tensor is a 4×4 matrix that defines distances in spacetime:

```
g_μν = diag(-e^(λ-μ), -ρ²e^(-μ), -e^μ, +e^(λ-μ))
```

| Component | Expression | Physical Role |
|-----------|------------|---------------|
| g₀₀ = g_ρρ | -e^(λ-μ) | Radial distances |
| g₁₁ = g_ΦΦ | -ρ²e^(-μ) | Angular distances |
| g₂₂ = g_zz | -e^μ | Axial distances |
| g₃₃ = g_tt | +e^(λ-μ) | Time intervals |

**Signature**: (-,-,-,+) meaning 3 spatial + 1 time dimension

---

## Electromagnetic Potentials

The electromagnetic field is derived from a **4-potential** A_μ.

### Potential Components

| Potential | Symbol | Related to | Physical Meaning |
|-----------|--------|------------|------------------|
| `phi(ρ, t)` | φ | A_Φ (azimuthal) | Generates radial-azimuthal fields |
| `psi(ρ, t)` | ψ | A_z (axial) | Generates radial-axial fields |

### Three Solution Cases

| Case | Condition | Physical Interpretation |
|------|-----------|------------------------|
| **Case I** | φ = 0 | Only axial potential; azimuthal magnetic field |
| **Case II** | ψ = 0 | Only azimuthal potential; axial magnetic field |
| **Case III** | φ ≠ 0, ψ ≠ 0 | Both potentials; general EM field |

---

## Electromagnetic Field Tensor F_μν

The **field tensor** contains the electric and magnetic fields:

```
F_μν = A_μ,ν - A_ν,μ  (antisymmetric)
```

### Non-zero Components

| Component | Expression | Related Field |
|-----------|------------|---------------|
| F₀₁ = F_ρΦ | -∂φ/∂ρ / √(8π) | B_z (axial magnetic) |
| F₀₂ = F_ρz | -∂ψ/∂ρ / √(8π) | B_Φ (azimuthal magnetic) |
| F₁₃ = F_Φt | +∂φ/∂t / √(8π) | E_Φ (azimuthal electric) |
| F₂₃ = F_zt | +∂ψ/∂t / √(8π) | E_z (axial electric) |

### Physical Interpretation

For **Case II** (ψ = 0, only φ):
- F₀₁ ≠ 0 → **Axial magnetic field B_z**
- F₁₃ ≠ 0 if φ depends on t → **Azimuthal electric field E_Φ**

For **Case I** (φ = 0, only ψ):
- F₀₂ ≠ 0 → **Azimuthal magnetic field B_Φ**
- F₂₃ ≠ 0 if ψ depends on t → **Axial electric field E_z**

---

## Current Density J^μ

The **4-current density** represents charge and current distributions.

### Maxwell's Equations

```
∇_μ F^μν = 4π J^ν
```

In curved spacetime, the covariant divergence is:

```
∇_μ F^μν = (1/√(-g)) ∂_μ (√(-g) F^μν)
```

### Current Components

| Component | Symbol | Physical Meaning |
|-----------|--------|------------------|
| J^ρ | J_rho | Radial current density |
| J^Φ | J_Phi | Azimuthal current density |
| J^z | J_z | **Axial current density** (most common) |
| J^t | J_t | Charge density (times c) |

### Vacuum Solutions

For the exact solutions from the paper: **J^μ = 0 everywhere**

These are "in vacuo" solutions - electromagnetic fields exist without sources.
The `verify_vacuum()` method confirms this by computing ∇_μ F^μν ≈ 0.

---

## Surface Currents K^μ

When a field is confined to a region ρ < R, **surface currents** at the boundary sustain it.

### Jump Condition

```
K^μ = (1/4π) n_α [F^αμ]
```

where:
- n_α = outward unit normal to surface
- [F^αμ] = discontinuity in field across boundary

### Surface Current Components

| Component | Symbol | Physical Meaning |
|-----------|--------|------------------|
| K^ρ | K_rho | Should be ~0 (current tangent to surface) |
| K^Φ | K_Phi | **Azimuthal surface current** |
| K^z | K_z | **Axial surface current** |
| K^t | K_t | Surface charge density |

### Physical Picture

For a cylinder with axial magnetic field B_z (Case II):
- **Azimuthal surface currents K^Φ** at ρ = R sustain the field
- Similar to windings of a solenoid

For a cylinder with azimuthal magnetic field B_Φ (Case I):
- **Axial surface currents K^z** at ρ = R sustain the field
- Similar to a straight wire

### Total Currents

```python
I_z = ∮ K^z R dΦ = 2πR K^z    # Total axial current
I_Φ = ∫ K^Φ dz                 # Azimuthal current per unit length
```

---

## Energy-Momentum Tensor E^β_α

The **electromagnetic stress-energy tensor** describes energy and momentum in the field.

### Key Properties

1. **Trace-free**: E^α_α = 0 (electromagnetic fields have zero trace)
2. **Symmetric**: E^αβ = E^βα

### Components

| Component | Physical Meaning |
|-----------|------------------|
| E^t_t | Energy density |
| E^ρ_ρ | Radial stress |
| E^Φ_Φ | Azimuthal stress |
| E^z_z | Axial stress |
| E^ρ_t | Energy flux (Poynting vector) |

---

## Solution Variants

### Case II Variants

| Variant | φ dependence | Physical Character |
|---------|--------------|-------------------|
| `rho_only` | φ(ρ) | Static field, varies radially |
| `t_only` | φ(t) | Time-dependent, uniform in ρ |
| `rho_quadratic` | φ = ½aρ² + b | Quadratic radial profile |
| `t_linear` | φ = at + b | Linear time evolution |

### Parameters

| Parameter | Symbol | Role |
|-----------|--------|------|
| m | m | Field strength/scale |
| n | n | Phase/offset |
| a | a | Coupling constant |
| p, q | p, q | Integration constants |
| α, β | alpha, beta | General relation parameters |

---

## Units

The library uses **geometric units** (G = c = 1):

| Quantity | Geometric Units | SI Conversion |
|----------|-----------------|---------------|
| Length | meters | - |
| Time | meters | × c |
| Mass | meters | × G/c² |
| Charge | meters | × √(G/(4πε₀)) |
| Current | dimensionless | × √(G/(4πε₀))/c |

---

## Example: Physical Interpretation of Case II

```python
sol = CaseIISolution(variant='rho_only', m=1.0, n=0.5, a=1.0)
```

**What this represents**:

1. **Spacetime**: Curved by the electromagnetic field
   - μ(ρ) controls how z-distances vary with ρ
   - λ(ρ) controls overall curvature

2. **EM Field**: Axial magnetic field B_z
   - φ(ρ) generates F₀₁ (radial-azimuthal component)
   - Magnetic field lines point along z-axis
   - Field strength varies with ρ

3. **Sources**: None (vacuum solution)
   - J^μ = 0 everywhere
   - Field is self-sustaining in curved spacetime

4. **Boundary**: If truncated at ρ = R
   - Azimuthal surface current K^Φ required
   - Acts like solenoid windings

---

## Junction Conditions (Israel-Darmois Formalism)

When matching two spacetime regions at a boundary ρ = R, the **Israel-Darmois junction conditions** ensure physical consistency.

### The Hypersurface Σ

A cylindrical hypersurface at ρ = R has:
- **Tangent coordinates**: (Φ, z, t)
- **Normal direction**: ρ
- **Signature**: (-,-,+) — timelike surface

### Induced Metric (First Fundamental Form)

The metric inherited by Σ from the ambient spacetime:

| Component | Expression | Meaning |
|-----------|------------|---------|
| γ_ΦΦ | -R² e^(-μ) | Angular distances |
| γ_zz | -e^μ | Axial distances |
| γ_tt | e^(λ-μ) | Time intervals |

### Extrinsic Curvature (Second Fundamental Form)

Measures how Σ bends within 4D spacetime:

| Component | Expression | Meaning |
|-----------|------------|---------|
| K_ΦΦ | R e^((-μ-λ)/2) (1 - ½R μ_ρ) | Azimuthal bending |
| K_zz | ½ e^((3μ-λ)/2) μ_ρ | Axial bending |
| K_tt | -½ e^((λ-μ)/2) (λ_ρ - μ_ρ) | Temporal bending |

### Junction Conditions

**First Condition (Darmois)**: [γ_ab] = 0
- Induced metric continuous across Σ
- No gaps or overlaps in spacetime

**Second Condition (Israel)**: [K_ab] = -8π(S_ab - ½γ_ab S)
- K_ab jump related to surface stress-energy S_ab
- If [K_ab] = 0: smooth junction, no shell
- If [K_ab] ≠ 0: thin shell of matter required

### Thin Shell Properties

| Quantity | Symbol | Physical Meaning |
|----------|--------|------------------|
| Surface energy density | σ = S^t_t | Energy per unit area |
| Azimuthal pressure | p_Φ = -S^Φ_Φ | Tangential stress |
| Axial pressure | p_z = -S^z_z | Tangential stress |

### Energy Conditions

| Condition | Requirement | Physical Meaning |
|-----------|-------------|------------------|
| Weak Energy | σ ≥ 0 | Positive energy |
| Dominant Energy | σ ≥ \|p_i\| | Energy dominates pressure |

---

## Wave Mode Conversion

The **wave conversion module** studies nonlinear conversion between gravitational and electromagnetic waves, based on Mishima & Tomizawa (2024).

### C-Energy (Thorne 1965)

A local, conserved energy measure for cylindrical waves:

| Mode | Symbol | Physical Meaning |
|------|--------|------------------|
| + polarization | ℰ_+ | Gravitational wave |
| × polarization | ℰ_× | Gravitational wave |
| z-mode | ℰ_z | Electromagnetic (axial) |
| φ-mode | ℰ_φ | Electromagnetic (azimuthal) |

### Occupancy Ratios

```
R_grav = (ℰ_+ + ℰ_×) / ℰ_total
R_em = (ℰ_z + ℰ_φ) / ℰ_total
```

With R_grav + R_em = 1 (energy conservation).

### Mode Conversion Physics

Near the symmetry axis (small ρ):
- Strong gravitational fields enable nonlinear coupling
- Energy transfers between GW and EM modes
- Purely general relativistic effect (no Newtonian analogue)

---

## References

- Misra, M. & Radhakrishna, L. (1962). "Some Electromagnetic Fields of Cylindrical Symmetry." *Proc. Nat. Inst. Sci. India* 28A(4), 632-645.
- Einstein, A. & Rosen, N. (1937). "On Gravitational Waves." *J. Franklin Inst.* 223, 43.
- Bonnor, W.B. (1954). "Static Magnetic Fields in General Relativity." *Proc. Phys. Soc. A* 67, 225.
- Israel, W. (1966). "Singular hypersurfaces and thin shells in general relativity." *Nuovo Cimento B* 44, 1-14.
- Mishima, T. & Tomizawa, S. (2024). "Nonlinear dynamics driving the conversion of gravitational and electromagnetic waves." *Phys. Rev. D* 110, 024038.
- Thorne, K.S. (1965). "Energy of Infinitely Long, Cylindrically Symmetric Systems." *Phys. Rev.* 138, B251.
