# Project: Leacock-Padgett Quantum Hamilton-Jacobi Solver — Step 6

## Goal

Push LP+PINN into 6D where grid-based methods become impractical. The question shifts from "is LP+PINN faster?" to "can LP+PINN produce answers where DVR cannot?"

## Background

At 6D, the computational landscape changes:

| Grid points/dim | 4D basis | 5D basis | 6D basis | Memory (dense) |
|-----------------|----------|----------|----------|----------------|
| 10 | 10,000 | 100,000 | 1,000,000 | 8 TB |
| 15 | 50,625 | 759,375 | 11,390,625 | 1000+ TB |
| 8 | 4,096 | 32,768 | 262,144 | 550 GB |

Even sparse DVR struggles beyond 5D. Alternative methods (MCTDH, DMC) become the only options—and LP+PINN might compete with these.

## Test Systems

### System A: 6D Separable Oscillator (Validation)

```
V(x₁, x₂, x₃, x₄, x₅, x₆) = ½(x₁² + x₂² + x₃² + x₄² + x₅² + x₆²)
```

**Exact ground state:** E = 3.0 (six zero-point contributions of 0.5 each)

**Exact momentum:** pᵢ = -ixᵢ for all i

This validates the 6D implementation before tackling coupled systems.

### System B: 6D Coupled Oscillator

```
V = ½Σᵢxᵢ² + λΣᵢxᵢxᵢ₊₁
```

With λ = 0.15 (nearest-neighbor coupling, periodic: x₆ couples to x₁).

**Reference:** No DVR possible. Compare against:
1. Perturbation theory (weak coupling limit)
2. Normal mode analysis (exact for harmonic coupling)
3. Variational bound (PINN energy should be ≥ true ground state)

### System C: Triatomic Vibrational Model (Physical System)

A simplified model of a linear triatomic molecule (like CO₂ or HCN):

```
V(r₁, r₂, θ₁, θ₂, φ₁, φ₂) = V_stretch(r₁, r₂) + V_bend(θ₁, θ₂) + V_coupling
```

Simplified version with harmonic terms:
```
V = ½k₁(r₁² + r₂²) + ½k₂(θ₁² + θ₂²) + ½k₃(φ₁² + φ₂²) + λ(r₁θ₁ + r₂θ₂)
```

This represents stretch-bend coupling in a triatomic.

**Reference:** Literature values for vibrational frequencies of real molecules, or MCTDH calculations.

## Architecture for 6D

```
Input: (x₁, x₂, x₃, x₄, x₅, x₆, quantum_numbers) — 12+ dimensions
Output: (p₁, p₂, p₃, p₄, p₅, p₆) — 12 real values

Hidden layers: 8 layers, 512-1024 neurons each
Parameters: ~4-8 million
```

### Memory Comparison

| Method | 6D Memory |
|--------|-----------|
| Dense DVR (n=10) | ~8 TB |
| Sparse DVR (n=10) | ~50-100 GB |
| LP+PINN (8 layers, 512 neurons) | ~20 MB |

The PINN memory advantage is now 1000x or more.

### Pole Structure for 6D

For state (n₁, n₂, n₃, n₄, n₅, n₆):

```
pᵢ = NNᵢ(x₁,...,x₆) + Σⱼ 1/(xᵢ - x₀ᵢⱼ)
```

Where the number of poles in direction i equals nᵢ.

Ground state (0,0,0,0,0,0) has no poles—pure neural network.

## Loss Function

Same structure, extended to 6D:

**Physics residual:**
```
L_physics = |Σᵢpᵢ² + iℏΣᵢ(∂pᵢ/∂xᵢ) - 2m(E - V)|²
```

**Curl-free constraints** (15 pairs in 6D):
```
L_curl = Σᵢ<ⱼ |∂pᵢ/∂xⱼ - ∂pⱼ/∂xᵢ|²
```

**Quantization loss:**
```
L_quant = Σᵢ |Jᵢ/ℏ - (nᵢ + 0.5)|²
```

## Training Strategy for 6D

Training in 6D requires more care:

### Collocation Points

- 6D space is vast; random sampling is inefficient
- Use importance sampling concentrated near:
  - The classical turning surface (where E = V)
  - The origin (where ground state density peaks)
  - Along coordinate axes (for symmetry)

Recommended: 50,000 - 100,000 collocation points, refreshed each epoch.

### Curriculum Learning

1. **Phase 1 (1 hour):** Train on separable potential (λ=0) to learn basic structure
2. **Phase 2 (1 hour):** Gradually increase coupling λ from 0 to target value
3. **Phase 3 (1 hour):** Fine-tune at target coupling with full loss

### Learning Rate Schedule

```
Epoch 1-1000: lr = 1e-3
Epoch 1000-5000: lr = 5e-4
Epoch 5000-15000: lr = 1e-4
Epoch 15000+: lr = 1e-5
```

### Checkpointing

Save model every 30 minutes. 6D training may need to be resumed.

## Validation Protocol

### 6D Separable (Ground State)

Since DVR is impractical, validate against exact analytical results:

| Metric | Target |
|--------|--------|
| Energy | 3.0 ± 1% |
| All pᵢ at (1,1,1,1,1,1) | -i ± 5% |
| Physics residual | < 10⁻³ |
| Curl residual | < 10⁻³ |

### 6D Coupled (Ground State)

**Perturbation theory prediction** for small λ:
```
E ≈ 3.0 - λ²(correction term)
```

For λ = 0.15 with nearest-neighbor periodic coupling, compute the perturbative correction and verify LP+PINN is within 5%.

**Normal mode exact solution:**
The coupled harmonic oscillator has exact solution via diagonalization of the coupling matrix. Compute analytically and compare.

**Variational bound:**
LP+PINN energy must be ≥ exact ground state energy (within numerical tolerance).

### Spot-Check Values (6D Separable Ground State)

| Point | Exact p₁ | Exact p₂ | ... | Exact p₆ |
|-------|----------|----------|-----|----------|
| (1,0,0,0,0,0) | -i | 0 | ... | 0 |
| (1,1,1,1,1,1) | -i | -i | ... | -i |
| (0.5,1,1.5,2,2.5,3) | -0.5i | -i | -1.5i | -2i | -2.5i | -3i |

## Reference Methods (No DVR)

Since DVR is impractical, implement alternative references:

### 1. Normal Mode Analysis (Exact for Coupled Harmonic)

```python
# For V = ½x'Ax where A is the coupling matrix
eigenvalues = np.linalg.eigvalsh(A)
omega = np.sqrt(eigenvalues)
E_ground = 0.5 * np.sum(omega)
```

This gives the exact ground state energy for any quadratically coupled system.

### 2. Perturbation Theory

```python
# First-order: E₁ = <0|V'|0> = 0 for off-diagonal coupling
# Second-order: E₂ = -Σₙ |<n|V'|0>|² / (Eₙ - E₀)
```

### 3. Variational Monte Carlo (Optional)

If you have VMC infrastructure, run a simple Gaussian ansatz for comparison.

## Computational Targets

### 6D Separable

| Metric | Target |
|--------|--------|
| Energy error | < 1% |
| Training time | < 3 hours |
| Memory | < 100 MB |

### 6D Coupled

| Metric | Target |
|--------|--------|
| Energy error vs normal mode exact | < 2% |
| Training time | < 4 hours |
| Memory | < 100 MB |

## Deliverables

1. 6D PINN architecture with 6-component momentum output
2. Importance sampling for 6D collocation points
3. Normal mode reference solver for coupled harmonic systems
4. Energy results for 6D separable and coupled oscillators
5. Comparison: LP+PINN vs normal mode exact vs perturbation theory
6. Training curves showing convergence over 3+ hours
7. Memory usage verification (should be << 1 GB)

## Success Criteria

### Minimum Success (Method Works)

- 6D separable ground state within 2% of exact
- 6D coupled ground state within 5% of normal mode exact
- Training completes in < 4 hours
- Memory stays below 500 MB

### Strong Success (Method is Useful)

- 6D separable within 1%
- 6D coupled within 2%
- Can compute first excited state (1,0,0,0,0,0)
- Results are reproducible across random seeds

### Exceptional Success (Novel Capability)

- Accurate results for triatomic vibrational model
- Can compare meaningfully against literature MCTDH results
- Demonstrates clear scaling advantage: 6D takes similar time to 4D

## Why This Matters

Step 6 answers the ultimate question: can LP+PINN solve problems that grid methods cannot?

At 6D with coupled potentials, we're beyond what standard DVR can handle. If LP+PINN produces accurate results here—validated against normal mode analysis for the harmonic case—it demonstrates genuine capability in a regime where few methods work.

The triatomic model, if successful, connects to real physical systems and could point toward actual applications in molecular spectroscopy.

## Notes on Expected Difficulty

Be prepared for:

1. **Slower convergence:** 6D optimization landscape is harder; expect 2-3x more epochs than 4D
2. **More hyperparameter sensitivity:** Learning rate and network size matter more
3. **Validation challenges:** Without DVR, you rely on analytical solutions and consistency checks
4. **Possible failure:** If 6D doesn't converge, this is a legitimate finding about the method's limits

Document everything, including failures. Negative results are valuable.
