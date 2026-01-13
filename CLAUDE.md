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

## References

- Misra, M. & Radhakrishna, L. (1962). "Some Electromagnetic Fields of Cylindrical Symmetry." *Proc. Nat. Inst. Sci. India* 28A(4), 632-645.
- Mishima, T. & Tomizawa, S. (2024). "Nonlinear dynamics driving the conversion of gravitational and electromagnetic waves in cylindrically symmetric spacetime." *Phys. Rev. D* 110, 024038.
- Israel, W. (1966). "Singular hypersurfaces and thin shells in general relativity." *Nuovo Cimento B* 44, 1-14.

## Documentation

- `PHYSICS_GUIDE.md` - Physical interpretation of quantities
- `WAVE_CONVERSION.md` - GW-EM wave mode conversion module
- `JUNCTION_CONDITIONS.md` - Israel-Darmois junction conditions module
