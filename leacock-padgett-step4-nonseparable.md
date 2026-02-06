# Project: Leacock-Padgett Quantum Hamilton-Jacobi Solver — Step 4

## Goal

Apply the PINN-based LP solver to 2D non-separable systems where no analytical solutions exist. Validate against established numerical quantum methods.

## Background

Non-separable potentials couple the degrees of freedom, so V(x, y) ≠ V_x(x) + V_y(y). The quantum Hamilton-Jacobi equation remains the same:

```
p_x² + p_y² + iℏ(∂p_x/∂x + ∂p_y/∂y) = 2m(E - V(x, y))
```

But now p_x depends on y and vice versa. The action variables J_x and J_y are no longer independent, and quantization becomes a 2D root-finding problem.

## Test Systems

### System A: Coupled Harmonic Oscillators

```
V(x, y) = ½(x² + y²) + λxy
```

With λ = 0.2 (weak coupling) and λ = 0.5 (moderate coupling).

**Reference energies** (compute via DVR or basis set expansion):

For λ = 0.2:
| State | DVR Energy | 
|-------|------------|
| Ground | ~0.9165 |
| 1st excited | ~1.8330 |
| 2nd excited | ~1.9165 |

For λ = 0.5:
| State | DVR Energy |
|-------|------------|
| Ground | ~0.8660 |
| 1st excited | ~1.7321 |
| 2nd excited | ~1.8660 |

Note: Compute your own reference values with a DVR implementation. These are approximate.

### System B: Hénon-Heiles Potential

```
V(x, y) = ½(x² + y²) + λ(x²y - y³/3)
```

Classic test case for quantum chaos. Use λ = 0.1 (regular regime).

**Reference energies** (from literature or DVR):

| State | Approximate Energy |
|-------|-------------------|
| Ground | ~1.00 |
| 1st excited | ~1.99 |
| 2nd excited | ~2.00 |
| 3rd excited | ~2.97 |

Higher states become increasingly affected by chaos; stick to the first 5-10 states.

## Implementation Changes

### Energy as Network Input

For non-separable systems, scanning over E to find quantized states is expensive if you retrain for each E. Modify the architecture:

```
Input: (x, y, E) — position and energy
Output: (p_x, p_y) — momentum field at that energy

Hidden layers: 5-6 layers, 256 neurons each
```

Train on a range of energies simultaneously, then extract the quantization condition.

### Quantization Condition

For non-separable systems, the simple J_x = (n_x + ½)ℏ condition doesn't directly apply. Instead:

1. **EBK quantization**: Find energies where both action integrals simultaneously satisfy quantization conditions along topologically independent loops.

2. **Practical approach**: Scan over E values, compute both J_x(E) and J_y(E), find E where both are half-integers.

For coupled oscillators with weak coupling, the quantization conditions remain approximately:
```
J_x ≈ (n_x + ½)ℏ
J_y ≈ (n_y + ½)ℏ
```

But J_x and J_y now depend on both n_x and n_y implicitly through E.

### Contour Selection

In non-separable systems, the classical turning points form curves, not isolated points. The contour for J_x should:

1. Encircle the region where motion in x is classically allowed (at fixed y)
2. Remain in a region where the momentum field is well-defined

This may require adaptive contour selection based on the potential landscape.

## Loss Function

Same as Step 3, with increased importance on the curl constraint:

```
L = L_physics + λ₁ * L_curl + λ₂ * L_boundary
```

The curl constraint (∂p_x/∂y = ∂p_y/∂x) is critical—non-separable doesn't mean non-gradient.

## Validation Protocol

Since we lack analytical solutions:

### 1. Implement DVR Reference Solver

Create a Discrete Variable Representation solver for 2D:

```python
# Grid-based quantum solver
# Use 50-100 grid points per dimension
# Diagonalize the Hamiltonian matrix
# Extract lowest 10-20 eigenvalues
```

This gives "ground truth" energies to compare against.

### 2. Compare LP+PINN vs DVR

| Metric | Target |
|--------|--------|
| Ground state energy error | < 1% |
| First 5 excited state errors | < 2% |
| Correct state ordering | Must match DVR |

### 3. Consistency Checks

- **Coupling limit**: As λ → 0, energies should approach separable case
- **Symmetry**: For symmetric potentials, degenerate states should have equal energies
- **Variational bound**: LP energies should not be below DVR energies (both are exact in principle, but numerical errors differ)

## Spot-Check Protocol

Unlike Steps 1-3, we don't have exact p(x, y) to compare against. Instead verify:

**Physics residual at random points:**

Sample 100 random points in the classically allowed region. Evaluate:
```
residual = |p_x² + p_y² + iℏ(∂p_x/∂x + ∂p_y/∂y) - 2m(E - V)|
```

All residuals should be < 10⁻⁴.

**Curl check:**
```
curl = |∂p_x/∂y - ∂p_y/∂x|
```

Should be < 10⁻⁴ everywhere.

**Asymptotic behavior:**

In classically forbidden regions (where E < V), verify:
```
|p| → √(2m(V - E))  (purely imaginary momentum)
```

## Computational Notes

- DVR with 80×80 grid is ~6400 basis functions, manageable on CPU
- PINN training will take 30-60 minutes per energy on GPU
- For energy scanning, train on E ∈ [0.5, 5.0] with 50-100 energy samples
- Hénon-Heiles becomes chaotic above E ~ 1/6λ²; stay below this

## Deliverables

1. DVR reference solver for 2D potentials
2. Modified PINN architecture with E as input
3. Energy eigenvalue comparison: LP+PINN vs DVR for both test systems
4. Residual and curl diagnostics
5. Plot: J_x(E) and J_y(E) showing quantization crossings
6. Performance metrics: training time, accuracy vs DVR

## Success Criteria

**Pass**: LP+PINN energies match DVR to within 2% for first 5 states on both test systems.

**Fail indicators**:
- Energies systematically above or below DVR
- Missing states or extra spurious states
- Large physics residuals despite good energy agreement (overfitting)

## Why This Matters

This is the first step where we're doing something that isn't just reproducing known results a different way. If LP+PINN matches DVR here, we have evidence the method works beyond toy problems. Step 5 will push into dimensions where DVR becomes expensive, and the comparison becomes more interesting.
