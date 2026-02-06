# Project: Leacock-Padgett Quantum Hamilton-Jacobi Solver — Step 3

## Goal

Extend the PINN-based LP solver to 2D separable systems, validating against exact solutions before moving to non-separable problems.

## Background

For a 2D system with potential V(x, y), the quantum Hamilton-Jacobi equation becomes:

```
p_x² + p_y² + iℏ(∂p_x/∂x + ∂p_y/∂y) = 2m(E - V(x, y))
```

where p_x = ∂W/∂x and p_y = ∂W/∂y are components of the quantum momentum field.

For separable potentials V(x, y) = V_x(x) + V_y(y), the problem decomposes into two 1D problems:

```
E = E_x + E_y
J_x = (n_x + ½)ℏ
J_y = (n_y + ½)ℏ
```

This lets us validate the 2D implementation against known results.

## Step 3 Task

Implement a 2D PINN solver and validate on the 2D isotropic harmonic oscillator:

```
V(x, y) = ½m(ω_x²x² + ω_y²y²)
```

### Part A: Separable case (ω_x = ω_y = 1)

**Exact energies (natural units):**

```
E_{n_x, n_y} = (n_x + ½) + (n_y + ½) = n_x + n_y + 1
```

| State | (n_x, n_y) | Energy | Degeneracy |
|-------|------------|--------|------------|
| Ground | (0, 0) | 1.0 | 1 |
| First excited | (1, 0), (0, 1) | 2.0 | 2 |
| Second excited | (2, 0), (1, 1), (0, 2) | 3.0 | 3 |
| Third excited | (3, 0), (2, 1), (1, 2), (0, 3) | 4.0 | 4 |

### Part B: Non-degenerate case (ω_x = 1, ω_y = √2)

**Exact energies:**

```
E_{n_x, n_y} = (n_x + ½) + √2(n_y + ½)
```

| State | (n_x, n_y) | Energy |
|-------|------------|--------|
| (0, 0) | 0.5 + 0.7071 | 1.2071 |
| (1, 0) | 1.5 + 0.7071 | 2.2071 |
| (0, 1) | 0.5 + 2.1213 | 2.6213 |
| (2, 0) | 2.5 + 0.7071 | 3.2071 |
| (1, 1) | 1.5 + 2.1213 | 3.6213 |

## Architecture

```
Input: (x, y) — 4 real channels (Re(x), Im(x), Re(y), Im(y))
Output: (p_x, p_y) — 4 real channels

Hidden layers: 4-5 layers, 128-256 neurons each
Activation: tanh or sin
```

## Loss Function

**Physics residual:**

```
L_physics = mean(|p_x² + p_y² + iℏ(∂p_x/∂x + ∂p_y/∂y) - 2m(E - V)|²)
```

**Irrotationality constraint** (p is a gradient field):

```
L_curl = mean(|∂p_x/∂y - ∂p_y/∂x|²)
```

This enforces that p = ∇W for some scalar W, which is required by the LP formalism.

**Boundary loss:**

```
L_boundary = asymptotic behavior in classically forbidden regions
```

**Total loss:**

```
L = L_physics + λ₁ * L_curl + λ₂ * L_boundary
```

## Quantization in 2D

The action variables become line integrals around topologically independent loops:

```
J_x = (1/2πi) ∮ p_x dx  (loop in complex x-plane, y fixed)
J_y = (1/2πi) ∮ p_y dy  (loop in complex y-plane, x fixed)
```

For separable systems, these can be computed independently. The quantization conditions are:

```
J_x = (n_x + ½)ℏ
J_y = (n_y + ½)ℏ
```

## Training Procedure

1. **Fix total energy E** (e.g., E = 1.0 for ground state of isotropic oscillator)
2. **Sample collocation points** in 4D space (Re(x), Im(x), Re(y), Im(y))
   - Real parts: [-4, 4]
   - Imaginary parts: [-0.5, 0.5]
3. **Train network** to minimize total loss
4. **Compute J_x and J_y** via contour integration
5. **Verify** both equal (n + ½)ℏ for appropriate integers

## Validation Criteria

### Part A (isotropic):

| State | Target E | J_x/ℏ | J_y/ℏ | Tolerance |
|-------|----------|-------|-------|-----------|
| (0,0) | 1.0 | 0.5 | 0.5 | 10⁻³ |
| (1,0) | 2.0 | 1.5 | 0.5 | 10⁻³ |
| (0,1) | 2.0 | 0.5 | 1.5 | 10⁻³ |
| (1,1) | 3.0 | 1.5 | 1.5 | 10⁻³ |

### Part B (anisotropic):

| State | Target E | J_x/ℏ | J_y/ℏ | Tolerance |
|-------|----------|-------|-------|-----------|
| (0,0) | 1.2071 | 0.5 | 0.5 | 10⁻³ |
| (1,0) | 2.2071 | 1.5 | 0.5 | 10⁻³ |
| (0,1) | 2.6213 | 0.5 | 1.5 | 10⁻³ |

## Spot-Check Values

**Ground state (0,0) of isotropic oscillator, E = 1.0:**

Exact momentum field:
```
p_x(x, y) = -ix
p_y(x, y) = -iy
```

| Point (x, y) | Exact p_x | Exact p_y |
|--------------|-----------|-----------|
| (0, 0) | 0 | 0 |
| (1, 0) | -i | 0 |
| (0, 1) | 0 | -i |
| (1, 1) | -i | -i |
| (0.5+0.2i, 0) | -0.2 - 0.5i | 0 |

**First excited state (1,0), E = 2.0:**

```
p_x(x, y) = -ix + 1/x
p_y(x, y) = -iy
```

Note the pole at x = 0 (the nodal line of the wavefunction).

## Computational Notes

- Collocation point count scales as O(N²) for 2D; use 1000-5000 points
- Training will be slower than 1D; expect 10-30 minutes per state on GPU
- The curl constraint is critical—without it, the network can learn non-physical momentum fields
- For degenerate states, the network may find linear combinations; this is fine as long as energies match

## Deliverable

A Python module with:

1. 2D PINN architecture with curl constraint
2. 2D contour integration for J_x and J_y
3. Validation on both isotropic and anisotropic harmonic oscillators
4. Spot-check verification at specific points
5. Performance comparison: training time, accuracy vs 1D

## Why This Matters

This is the last step with full analytical verification. Step 4 will tackle non-separable 2D systems (coupled oscillators, Hénon-Heiles) where exact solutions don't exist and we'll need to compare against numerical quantum methods instead.
