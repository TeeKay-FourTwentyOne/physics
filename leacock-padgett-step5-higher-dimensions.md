# Project: Leacock-Padgett Quantum Hamilton-Jacobi Solver — Step 5

## Goal

Extend the LP+PINN solver to 3-4 dimensional systems and begin quantifying whether the approach offers computational advantages over grid-based methods.

## Background

At 3-4 dimensions, DVR becomes expensive:

| Dimensions | Grid points/dim | Total basis size | DVR matrix size |
|------------|-----------------|------------------|-----------------|
| 2D | 80 | 6,400 | ~300 MB |
| 3D | 50 | 125,000 | ~120 GB |
| 3D | 30 | 27,000 | ~6 GB |
| 4D | 20 | 160,000 | ~200 GB |
| 4D | 15 | 50,625 | ~20 GB |

The PINN approach has no explicit grid—network size is independent of dimensionality. This is where LP+PINN might start winning.

## Test Systems

### System A: 3D Isotropic Harmonic Oscillator (Validation)

```
V(x, y, z) = ½(x² + y² + z²)
```

**Exact energies:**

```
E_{n_x, n_y, n_z} = n_x + n_y + n_z + 3/2
```

| State | (n_x, n_y, n_z) | Energy | Degeneracy |
|-------|-----------------|--------|------------|
| Ground | (0,0,0) | 1.5 | 1 |
| 1st excited | (1,0,0), etc. | 2.5 | 3 |
| 2nd excited | (2,0,0), (1,1,0), etc. | 3.5 | 6 |

Use this to validate the 3D implementation before tackling non-separable cases.

### System B: 3D Coupled Oscillator

```
V(x, y, z) = ½(x² + y² + z²) + λ₁xy + λ₂yz
```

With λ₁ = 0.2, λ₂ = 0.15 (asymmetric coupling).

**Reference:** Compute via sparse DVR or Lanczos iteration (full diagonalization is expensive).

### System C: 4D Separable Oscillator (Validation)

```
V(x, y, z, w) = ½(x² + y² + z² + w²)
```

**Exact energies:**

```
E = n_x + n_y + n_z + n_w + 2
```

Ground state E = 2.0. Use to validate 4D implementation.

### System D: 4D Coupled Oscillator

```
V(x, y, z, w) = ½(x² + y² + z² + w²) + λ(xy + zw)
```

With λ = 0.2.

**Reference:** Sparse eigensolvers (Lanczos/Arnoldi) or MCTDH if available.

## Architecture Changes

### 3D Momentum Field

```
Input: (x, y, z, E) or (x, y, z, n_x, n_y, n_z)
Output: (p_x, p_y, p_z)

Hidden layers: 6 layers, 256-512 neurons each
```

### 4D Momentum Field

```
Input: (x, y, z, w, quantum_numbers)
Output: (p_x, p_y, p_z, p_w)

Hidden layers: 6-8 layers, 512 neurons each
```

### Pole Structure

For state (n_x, n_y, n_z):
- n_x poles in the x-direction (learnable positions)
- n_y poles in the y-direction
- n_z poles in the z-direction

```
p_x = NN_x(x,y,z) + Σᵢ 1/(x - x₀ᵢ)
p_y = NN_y(x,y,z) + Σⱼ 1/(y - y₀ⱼ)
p_z = NN_z(x,y,z) + Σₖ 1/(z - z₀ₖ)
```

## Loss Function

Same structure as Step 4, extended to 3D/4D:

**Physics residual:**
```
L_physics = |p_x² + p_y² + p_z² + iℏ(∂p_x/∂x + ∂p_y/∂y + ∂p_z/∂z) - 2m(E - V)|²
```

**Curl-free constraints** (momentum is a gradient):
```
L_curl = |∂p_x/∂y - ∂p_y/∂x|² + |∂p_y/∂z - ∂p_z/∂y|² + |∂p_z/∂x - ∂p_x/∂z|²
```

**Quantization loss:**
```
L_quant = |J_x/ℏ - (n_x + 0.5)|² + |J_y/ℏ - (n_y + 0.5)|² + |J_z/ℏ - (n_z + 0.5)|²
```

## Validation Protocol

### 3D Isotropic Oscillator

| State | Target E | J_x/ℏ | J_y/ℏ | J_z/ℏ | Tolerance |
|-------|----------|-------|-------|-------|-----------|
| (0,0,0) | 1.5 | 0.5 | 0.5 | 0.5 | 1% |
| (1,0,0) | 2.5 | 1.5 | 0.5 | 0.5 | 1% |
| (1,1,0) | 3.5 | 1.5 | 1.5 | 0.5 | 2% |
| (1,1,1) | 4.5 | 1.5 | 1.5 | 1.5 | 2% |

### 3D Coupled Oscillator

| State | DVR Energy | LP+PINN Energy | Target Error |
|-------|------------|----------------|--------------|
| (0,0,0) | compute | compute | < 1% |
| (1,0,0) | compute | compute | < 2% |
| (0,1,0) | compute | compute | < 2% |
| (0,0,1) | compute | compute | < 2% |

### Spot-Check Values (3D Isotropic Ground State)

Exact momentum: p_x = -ix, p_y = -iy, p_z = -iz

| (x, y, z) | Exact p_x | Exact p_y | Exact p_z |
|-----------|-----------|-----------|-----------|
| (1, 0, 0) | -i | 0 | 0 |
| (1, 1, 0) | -i | -i | 0 |
| (1, 1, 1) | -i | -i | -i |
| (0.5, 0.3, 0.2) | -0.5i | -0.3i | -0.2i |

## Computational Comparison

This is the key new element in Step 5. For each test system, measure and report:

### Metrics to Track

| Metric | DVR | LP+PINN |
|--------|-----|---------|
| Memory usage (GB) | ? | ? |
| Time to solution (s) | ? | ? |
| Energy accuracy | ? | ? |
| Scalability | O(N^d) | O(?) |

### Comparison Protocol

1. **3D Separable:** Both methods should succeed easily. Compare wall-clock time.

2. **3D Coupled:** DVR with 30³ = 27,000 basis functions. Measure:
   - DVR: matrix construction time + diagonalization time
   - LP+PINN: training time + quantization time

3. **4D Separable:** DVR with 20⁴ = 160,000 basis functions (may need sparse methods). This is where PINN might win on memory.

4. **4D Coupled:** DVR becomes painful. Can LP+PINN find the ground state faster?

### Expected Crossover

If LP+PINN has any advantage, it should appear in 4D. The hypothesis is:

- DVR memory: O(N^d) where N = grid points per dimension
- PINN memory: O(P) where P = network parameters (dimension-independent)

A 512-neuron, 6-layer network has ~1.5M parameters regardless of dimensionality. DVR at 4D with N=20 has 160,000 basis functions requiring ~200GB for the full Hamiltonian matrix.

## Reference Solver Updates

### Sparse DVR

For 3D+, implement sparse matrix construction:

```python
from scipy.sparse import kron, diags
from scipy.sparse.linalg import eigsh

# Build sparse Hamiltonian
H_sparse = build_sparse_hamiltonian(V, grid_points, dimensions)

# Find lowest k eigenvalues via Lanczos
energies, states = eigsh(H_sparse, k=10, which='SA')
```

This allows DVR to scale to 3D with reasonable memory.

### MCTDH Reference (Optional)

If available, the Multi-Configuration Time-Dependent Hartree method provides an alternative reference for 4D+ systems. Not required but useful for cross-validation.

## Implementation Notes

### Memory Management

- Use float32 instead of float64 for PINN if memory is tight
- Batch collocation points (don't load all at once)
- For DVR, use sparse matrices exclusively in 3D+

### Training Stability

- 3D+ training may be slower to converge; increase patience
- Consider learning rate scheduling
- Monitor all three curl components; if any diverge, reduce learning rate

### Contour Integration in 3D

The action integrals become:
```
J_x = (1/2πi) ∮ p_x dx  (contour in complex x-plane, y and z fixed at real values)
```

Compute each J independently by fixing the other coordinates.

## Deliverables

1. 3D and 4D PINN architectures with multi-dimensional pole support
2. Sparse DVR reference solver for 3D systems
3. Energy comparison tables: LP+PINN vs DVR for all test systems
4. **Computational cost comparison**: memory, time, and scaling analysis
5. Plot: Accuracy vs training time for 2D, 3D, 4D (same target accuracy)
6. Recommendation: at what dimensionality (if any) does LP+PINN become preferable?

## Success Criteria

### Accuracy

| System | Target |
|--------|--------|
| 3D separable | < 1% error |
| 3D coupled, ground state | < 1% error |
| 3D coupled, excited states | < 2% error |
| 4D separable | < 1% error |
| 4D coupled, ground state | < 2% error |

### Computational

**Weak win:** LP+PINN uses less memory than DVR at 4D while achieving comparable accuracy.

**Strong win:** LP+PINN is faster AND uses less memory than sparse DVR at 4D.

**No win:** LP+PINN shows no computational advantage through 4D. (This is still a valid scientific finding.)

## Why This Matters

Step 5 answers the central question: does the LP reformulation offer computational advantages when combined with neural networks?

If LP+PINN wins at 4D, Step 6 pushes to 6D (triatomic molecules) where the comparison becomes more dramatic.

If LP+PINN shows no advantage, we've learned that the quantum momentum field is not more compressible than the wavefunction—a useful negative result that explains why this approach hasn't been pursued before.
