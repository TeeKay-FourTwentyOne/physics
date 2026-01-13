# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project implements exact solutions of the Einstein-Maxwell field equations with cylindrical symmetry, based on the 1962 paper by Misra & Radhakrishna. It provides both symbolic (SymPy) and numerical (NumPy) calculations for electromagnetic fields in curved spacetime.

## Commands

```bash
# Run demo
python examples/demo.py

# Run tests
python -m pytest tests/ -v

# Install dependencies
pip install sympy numpy scipy pytest
```

## Architecture

```
em_cylindrical/           # Main package
├── __init__.py          # Exports all public classes and functions
├── metric.py            # EinsteinRosenMetric class for spacetime metric
├── field_tensor.py      # Electromagnetic field tensor F_μν
├── energy_tensor.py     # Energy-momentum tensor E^β_α
├── wave_equation.py     # Cylindrical wave equation solutions (Bessel functions)
├── current.py           # Current density calculations
├── boundary.py          # Surface currents at cylindrical boundaries
├── sources.py           # Non-vacuum solutions with current sources
├── solutions/           # Exact solutions for three cases
│   ├── case_i.py        # φ = 0 (only ψ non-zero)
│   ├── case_ii.py       # ψ = 0 (only φ non-zero)
│   └── case_iii.py      # Both φ and ψ non-zero
├── wave_conversion/     # GW-EM mode conversion (Mishima & Tomizawa 2024)
│   ├── ernst.py         # Ernst potential calculations
│   ├── seeds.py         # Vacuum seed solutions
│   ├── harmonic_map.py  # Geodesic submanifold embeddings
│   ├── c_energy.py      # C-energy and mode decomposition
│   └── conversion.py    # Main WaveConversion class
└── junction/            # Israel-Darmois junction conditions
    ├── hypersurface.py  # Cylindrical hypersurface geometry
    ├── extrinsic_curvature.py  # Second fundamental form K_ab
    ├── junction_conditions.py  # Junction condition verification
    └── matched_solution.py     # Two-region matched solutions
```

## Key Concepts

**Einstein-Rosen Metric** (equation 1):
```
ds² = e^(λ-μ)(dt² - dρ²) - ρ²e^(-μ)dΦ² - e^μdz²
```

**Coordinates**: (ρ, Φ, z, t) where ρ is radial, Φ is angular, z is axial, t is time.

**4-Potential Components**: φ (related to Φ direction) and ψ (related to z direction).

**Three Solution Cases**:
- Case (i): φ = 0 - includes Bessel function wave solutions
- Case (ii): ψ = 0 - various parametric forms
- Case (iii): Both non-zero - most general electromagnetic fields

## Usage Pattern

### Basic Solutions

```python
from em_cylindrical import CaseISolution, CaseIISolution, CaseIIISolution

# Create a solution with parameters
sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

# Get values at a point (ρ, t)
psi = sol.psi(2.0, 1.0)
mu = sol.mu(2.0, 1.0)
lam = sol.lambda_(2.0, 1.0)

# Get metric tensor
g = sol.metric_at(2.0, 1.0)  # 4x4 numpy array

# Verify field equations
residuals = sol.verify_field_equations(2.0, 1.0)
```

### Wave Conversion (GW-EM Mode Conversion)

```python
from em_cylindrical.wave_conversion import (
    WaveConversion, LinearWaveSeed, ComplexLine
)

# Create wave conversion solution
seed = LinearWaveSeed(amplitude=0.5, wavenumber=1.0)
wc = WaveConversion(seed=seed, submanifold=ComplexLine(theta=0.1))

# Get mode occupancy ratios
c = wc.c_energy()
R_grav, R_em = c.occupancy_ratios(rho=2.0, t=1.0)
```

### Junction Conditions (Matching Spacetime Regions)

```python
from em_cylindrical import CaseISolution
from em_cylindrical.junction import MatchedSolution, IsraelDarmoisConditions

# Match interior and exterior solutions at R = 2.0
matched = MatchedSolution(interior_sol, exterior_sol, junction_radius=2.0)

# Verify junction conditions
result = matched.junction().verify_all(t=1.0)
print(f"Valid junction: {result['first_condition']['satisfied']}")
print(f"Thin shell: {result['second_condition']['thin_shell']}")
```

## API Reference

### Solution Classes (em_cylindrical.solutions)

| Class | Description | Key Parameters |
|-------|-------------|----------------|
| `CaseISolution` | φ = 0 solutions | `variant`: 't_only', 'rho_only', 'bessel' |
| `CaseIISolution` | ψ = 0 solutions | `variant`: 'rho_only', 't_only', 'rho_quadratic', 't_linear' |
| `CaseIIISolution` | Both φ,ψ ≠ 0 | `variant`: 'variant_71', 'variant_72' |

**Common Solution Methods:**
- `psi(rho, t)` → EM potential ψ
- `phi(rho, t)` → EM potential φ
- `mu(rho, t)` → Metric function μ
- `lambda_(rho, t)` → Metric function λ
- `metric_at(rho, t)` → 4×4 metric tensor g_μν
- `inverse_metric_at(rho, t)` → 4×4 inverse metric g^μν
- `field_tensor_at(rho, t)` → 4×4 field tensor F_μν
- `energy_tensor_at(rho, t)` → 4×4 energy tensor E^β_α
- `verify_field_equations(rho, t)` → Dict of residuals

### Metric Module (em_cylindrical.metric)

| Class/Function | Description |
|----------------|-------------|
| `EinsteinRosenMetric(lambda_expr, mu_expr)` | Symbolic metric from λ, μ |
| `compute_christoffel_symbols(metric)` | Γ^i_jk as dict |

**EinsteinRosenMetric Methods:**
- `metric_at(rho, t)` → 4×4 numpy array
- `inverse_metric_at(rho, t)` → 4×4 inverse metric
- `ds_squared(rho, t, drho, dPhi, dz, dt)` → Line element
- `lambda_at(rho, t)`, `mu_at(rho, t)` → Scalar values

### Field Tensor Module (em_cylindrical.field_tensor)

| Class/Function | Description |
|----------------|-------------|
| `ElectromagneticFieldTensor(phi_expr, psi_expr)` | F_μν from potentials |
| `levi_civita_4d(i, j, k, l)` → ε_ijkl symbol |

**ElectromagneticFieldTensor Methods:**
- `at(rho, t)` → 4×4 antisymmetric tensor
- `invariants(rho, t, g_inv)` → (I₁, I₂) scalar invariants

### Energy Tensor Module (em_cylindrical.energy_tensor)

| Class | Description |
|-------|-------------|
| `ElectromagneticEnergyTensor(phi, psi, mu, lambda)` | E^β_α from solution |

**Methods:**
- `at(rho, t)` → 4×4 tensor
- `trace(rho, t)` → Scalar (should be ~0 for EM)
- `energy_conditions(rho, t)` → Dict with 'weak_energy', 'trace_free'

### Wave Equation Module (em_cylindrical.wave_equation)

| Class/Function | Description |
|----------------|-------------|
| `CylindricalWaveSolution(modes)` | Solution to cylindrical wave eq |
| `create_standing_wave(k, A)` | J₀(kρ)cos(kt) wave |
| `create_superposition(k_values, A_values)` | Sum of modes |

**CylindricalWaveSolution Methods:**
- `x(rho, t)`, `x_1(rho, t)`, `x_4(rho, t)` → Function and derivatives
- `verify_wave_equation(rho, t)` → Residual (should be ~0)

### Current Module (em_cylindrical.current)

| Class | Description |
|-------|-------------|
| `CurrentDensity(solution)` | Computes J^μ from solution |

**Methods:**
- `J_at(rho, t)` → 4-vector current density
- `verify_vacuum(rho, t)` → Check J ≈ 0

### Boundary Module (em_cylindrical.boundary)

| Class/Function | Description |
|----------------|-------------|
| `CylindricalBoundary(solution, radius)` | Surface at ρ = R |
| `compute_solenoid_current(solution, R, t)` | Helper function |

**CylindricalBoundary Methods:**
- `surface_current(t)` → K^μ 4-vector
- `total_axial_current(t)` → Integrated I_z

### Sources Module (em_cylindrical.sources)

| Class | Description |
|-------|-------------|
| `UniformAxialCurrent(radius, j_z)` | J^z constant inside R |
| `SurfaceCurrent(radius, K_z, K_Phi)` | Delta-function at R |
| `CustomCurrent(j_func)` | User-defined J^μ |
| `SourcedSolution(solution, current)` | Solution with source |

### Wave Conversion Module (em_cylindrical.wave_conversion)

| Class | Description |
|-------|-------------|
| `WaveConversion(seed, submanifold)` | Main conversion class |
| `LinearWaveSeed(amplitude, wavenumber)` | Bessel function seed |
| `SolitonicSeed(p, q, l)` | Economou-Tsoubelis soliton |
| `ComplexLine(theta)` | GW/EM mixing submanifold |
| `LagrangianPlane()` | Nontrivial conversion |
| `CEnergy` | C-energy calculator |

**WaveConversion Methods:**
- `ernst_potentials(rho, t)` → (E, F) complex potentials
- `metric_functions(rho, t)` → Dict with psi, gamma, A_z, chi
- `to_misra_radhakrishna(rho, t)` → Dict with mu, lambda, phi, psi_MR
- `c_energy()` → CEnergy object
- `occupancy_ratios(rho, t)` via c_energy → (R_grav, R_em)

### Junction Module (em_cylindrical.junction)

| Class | Description |
|-------|-------------|
| `CylindricalHypersurface(radius)` | ρ = R surface geometry |
| `ExtrinsicCurvature(hypersurface, solution)` | K_ab calculator |
| `IsraelDarmoisConditions(interior, exterior, R)` | Junction checker |
| `MatchedSolution(interior, exterior, R)` | Two-region solution |

**Key Methods:**
- `CylindricalHypersurface.induced_metric(sol, t)` → 3×3 γ_ab
- `ExtrinsicCurvature.K_ab(t)` → 3×3 extrinsic curvature
- `IsraelDarmoisConditions.verify_all(t)` → Complete analysis
- `MatchedSolution.junction()` → IsraelDarmoisConditions object

## Testing

```bash
# Run all tests (176 tests)
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_core.py -v
python -m pytest tests/test_junction.py -v
python -m pytest tests/test_wave_conversion.py -v
```

## References

- Misra, M. & Radhakrishna, L. (1962). "Some Electromagnetic Fields of Cylindrical Symmetry." *Proc. Nat. Inst. Sci. India* 28A(4), 632-645.
- Mishima, T. & Tomizawa, S. (2024). "Nonlinear dynamics driving the conversion of gravitational and electromagnetic waves in cylindrically symmetric spacetime." *Phys. Rev. D* 110, 024038.
- Israel, W. (1966). "Singular hypersurfaces and thin shells in general relativity." *Nuovo Cimento B* 44, 1-14.
- Thorne, K.S. (1965). "Energy of Infinitely Long, Cylindrically Symmetric Systems." *Phys. Rev.* 138, B251.

## Documentation

- `PHYSICS_GUIDE.md` - Physical interpretation of quantities
- `WAVE_CONVERSION.md` - GW-EM wave mode conversion module
- `JUNCTION_CONDITIONS.md` - Israel-Darmois junction conditions module
