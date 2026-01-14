# EMCylindrical - Maple Implementation

Maple implementation of exact solutions to the Einstein-Maxwell field equations with cylindrical symmetry.

Based on: Misra, M. & Radhakrishna, L. (1962). "Some Electromagnetic Fields of Cylindrical Symmetry." *Proc. Nat. Inst. Sci. India* 28A(4), 632-645.

## Installation

1. Copy the `em_cylindrical_maple/` directory to your working location
2. In Maple, navigate to the directory and load the package:

```maple
currentdir("/path/to/em_cylindrical_maple"):
read("EMCylindrical.mpl"):
```

## Quick Start

```maple
# Load the package
currentdir("/path/to/em_cylindrical_maple"):
read("EMCylindrical.mpl"):

# Create a Case I Bessel solution
sol := CaseISolution:-Create(1.0, 0.5, "bessel", k_param=1.0, A_param=0.5):

# Evaluate at a point (rho=2, t=1)
psi_val := CaseISolution:-Psi(sol, 2.0, 1.0);
mu_val := CaseISolution:-Mu(sol, 2.0, 1.0);
lam_val := CaseISolution:-Lambda(sol, 2.0, 1.0);

# Get the 4x4 metric tensor
g := CaseISolution:-MetricAt(sol, 2.0, 1.0);

# Access symbolic expressions
exprs := CaseISolution:-GetSymbolicExpressions(sol);
print("Symbolic psi:", exprs["psi"]);

# Verify field equations (residuals should be ~0)
residuals := CaseISolution:-VerifyFieldEquations(sol, 2.0, 1.0);
```

## Package Structure

```
em_cylindrical_maple/
├── EMCylindrical.mpl           # Main package loader
├── lib/
│   ├── EinsteinRosenMetric.mpl # Metric tensor, Christoffel symbols
│   ├── FieldTensor.mpl         # Electromagnetic field tensor F_μν
│   ├── EnergyTensor.mpl        # Energy-momentum tensor E^β_α
│   └── WaveEquation.mpl        # Cylindrical wave solutions (Bessel)
├── solutions/
│   ├── CaseISolution.mpl       # φ=0 solutions (3 variants)
│   ├── CaseIISolution.mpl      # ψ=0 solutions (4 variants)
│   └── CaseIIISolution.mpl     # Both φ,ψ≠0 (4 variants)
├── helpers/
│   ├── CurrentDensity.mpl      # 4-current density J^μ
│   └── CylindricalBoundary.mpl # Surface currents at ρ=R
└── tests/
    └── test_core.mpl           # Test suite
```

## Modules

### EinsteinRosenMetric

Einstein-Rosen metric for cylindrically symmetric spacetimes:
```
ds² = e^(λ-μ)(dt² - dρ²) - ρ²e^(-μ)dΦ² - e^μdz²
```

```maple
metric := EinsteinRosenMetric:-Create(lambda_expr, mu_expr):
g := EinsteinRosenMetric:-MetricAt(metric, rho_val, t_val):
g_inv := EinsteinRosenMetric:-InverseMetricAt(metric, rho_val, t_val):
christoffel := EinsteinRosenMetric:-ComputeChristoffel(metric):
```

### FieldTensor

Electromagnetic field tensor F_μν from 4-potential components φ and ψ.

```maple
tensor := FieldTensor:-Create(phi_expr, psi_expr):
F := FieldTensor:-At(tensor, rho_val, t_val):
[I1, I2] := FieldTensor:-Invariants(tensor, rho_val, t_val, g_inv):
```

### EnergyTensor

Energy-momentum tensor E^β_α (source term in Einstein equations).

```maple
E_tensor := EnergyTensor:-Create(phi, psi, mu, lambda):
E := EnergyTensor:-At(E_tensor, rho_val, t_val):
trace := EnergyTensor:-Trace(E_tensor, rho_val, t_val):
```

### WaveEquation

Cylindrical wave equation solutions using Bessel functions.

```maple
wave := WaveEquation:-CreateStandingWave(k, A):
x_val := WaveEquation:-EvaluateX(wave, rho_val, t_val):
residual := WaveEquation:-VerifyWaveEquation(wave):  # Should be 0
```

### Solution Classes

#### CaseISolution (φ = 0)

Three variants:
- `"bessel"`: x = A·J₀(k·ρ)·cos(k·t + ε)
- `"t_only"`: ψ depends only on t (equation 43)
- `"rho_only"`: Metric depends on ρ (equation 44)

```maple
sol := CaseISolution:-Create(alpha, beta, "bessel",
    k_param=1.0, A_param=0.5, epsilon=0.0):
```

#### CaseIISolution (ψ = 0)

Four variants:
- `"rho_only"`: φ depends only on ρ (equation 56)
- `"t_only"`: φ depends only on t (equation 59)
- `"rho_quadratic"`: φ = ½aρ² + b (equation 57)
- `"t_linear"`: φ = at + b (equation 58)

```maple
sol := CaseIISolution:-Create(alpha, beta, "rho_only",
    m=1.0, n=0.0, a=1.0):
```

#### CaseIIISolution (φ ≠ 0, ψ ≠ 0)

Four variants:
- `"variant_71"`: ψ = at + b, φ depends on ρ
- `"variant_72"`: ψ = ½aρ² + b, φ depends on t
- `"variant_73"`: ψ depends on ρ (tanh), φ = at + b
- `"variant_78"`: φ quadratic in ρ (Bonnor's theorem)

```maple
sol := CaseIIISolution:-Create("variant_71", m=1.0, n=0.0, a=1.0):
```

### Common Solution Methods

All solution classes provide:

| Method | Description |
|--------|-------------|
| `Psi(sol, rho, t)` | Evaluate ψ potential |
| `Phi(sol, rho, t)` | Evaluate φ potential |
| `Mu(sol, rho, t)` | Evaluate μ metric function |
| `Lambda(sol, rho, t)` | Evaluate λ metric function |
| `MetricAt(sol, rho, t)` | 4×4 metric tensor g_μν |
| `InverseMetricAt(sol, rho, t)` | 4×4 inverse metric g^μν |
| `FieldTensorAt(sol, rho, t)` | 4×4 field tensor F_μν |
| `EnergyTensorAt(sol, rho, t)` | 4×4 energy tensor E^β_α |
| `GetSymbolicExpressions(sol)` | Table of symbolic expressions |
| `VerifyFieldEquations(sol, rho, t)` | Check field equation residuals |

## Running Tests

```maple
currentdir("/path/to/em_cylindrical_maple"):
read("EMCylindrical.mpl"):
read("tests/test_core.mpl"):
RunAllTests();
```

## Python/SymPy to Maple Translation

| Python (SymPy) | Maple |
|----------------|-------|
| `sp.symbols('rho')` | `rho := 'rho'` |
| `sp.diff(expr, x)` | `diff(expr, x)` |
| `sp.exp(x)` | `exp(x)` |
| `sp.log(x)` | `ln(x)` |
| `sp.sqrt(x)` | `sqrt(x)` |
| `sp.sin(x)`, `sp.cos(x)` | `sin(x)`, `cos(x)` |
| `sp.tanh(x)`, `sp.cosh(x)` | `tanh(x)`, `cosh(x)` |
| `sp.besselj(0, x)` | `BesselJ(0, x)` |
| `sp.bessely(0, x)` | `BesselY(0, x)` |
| `sp.pi` | `Pi` |
| `sp.Matrix([[...]])` | `Matrix([[...]])` |
| `M.det()` | `LinearAlgebra[Determinant](M)` |
| `sp.simplify(expr)` | `simplify(expr)` |
| `sp.Rational(1, 2)` | `1/2` |
| `expr.subs({x: val})` | `subs({x = val}, expr)` |
| `sp.lambdify((x,y), expr)` | `evalf(subs({x=xval, y=yval}, expr))` |

**Important**: Python uses 0-based indexing; Maple uses 1-based indexing.

## Correspondence with Python Package

This Maple package mirrors the Python `em_cylindrical` package:

| Python Module | Maple Module |
|---------------|--------------|
| `em_cylindrical.metric` | `EinsteinRosenMetric` |
| `em_cylindrical.field_tensor` | `FieldTensor` |
| `em_cylindrical.energy_tensor` | `EnergyTensor` |
| `em_cylindrical.wave_equation` | `WaveEquation` |
| `em_cylindrical.solutions.case_i` | `CaseISolution` |
| `em_cylindrical.solutions.case_ii` | `CaseIISolution` |
| `em_cylindrical.solutions.case_iii` | `CaseIIISolution` |
| `em_cylindrical.current` | `CurrentDensity` |
| `em_cylindrical.boundary` | `CylindricalBoundary` |

**Not ported** (numerical only, no symbolic computation):
- `em_cylindrical.wave_conversion/*` - GW-EM mode conversion
- `em_cylindrical.junction/*` - Israel-Darmois junction conditions

## References

- Misra, M. & Radhakrishna, L. (1962). "Some Electromagnetic Fields of Cylindrical Symmetry." *Proc. Nat. Inst. Sci. India* 28A(4), 632-645.
- See `PHYSICS_GUIDE.md` in the main repository for physical interpretation.
