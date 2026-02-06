# Project: Leacock-Padgett Quantum Hamilton-Jacobi Solver — Step 2

## Goal

Replace the numerical ODE solver with a Physics-Informed Neural Network (PINN) that learns the quantum momentum function p(x).

## Background

In Step 1, we solved the quantum Hamilton-Jacobi equation directly via numerical integration:

```
p² + iℏ(dp/dx) = 2m(E - V(x))
```

Now we train a neural network to represent p(x) such that it satisfies this equation. This is the foundation for scaling to higher dimensions where direct ODE integration becomes intractable.

## Step 2 Task

Implement a PINN-based solver for the 1D harmonic oscillator that:

1. Represents p(x; θ) as a neural network with parameters θ
2. Trains by minimizing a physics loss that enforces the quantum HJ equation
3. Computes the contour integral J from the learned p(x)
4. Recovers the same energy eigenvalues as the Step 1 solver

## Architecture

```
Input: x (complex-valued, or real + imaginary as 2 channels)
Output: p (complex-valued, or real + imaginary as 2 channels)

Hidden layers: 3-4 layers, 64-128 neurons each
Activation: tanh or sin (smooth, differentiable everywhere)
```

The network must handle complex inputs since the contour integral requires evaluating p(x) off the real axis.

## Loss Function

**Physics residual loss:**

```
L_physics = mean(|p² + iℏ(dp/dx) - 2m(E - V(x))|²)
```

Evaluate at collocation points sampled in the complex x-plane, concentrated near the real axis and around the contour path.

**Boundary loss (optional but helps convergence):**

```
L_boundary = |p(x_far) - p_asymptotic|²
```

For bound states, p(x) → ±i√(2m(V(x) - E)) as x → ±∞ (classically forbidden region behavior).

**Total loss:**

```
L = L_physics + λ * L_boundary
```

## Training Procedure

1. **Fix energy E** (start with E = 0.5 for ground state)
2. **Sample collocation points** in complex x-plane: real part in [-5, 5], imaginary part in [-1, 1]
3. **Train network** to minimize L until convergence (loss < 10⁻⁶)
4. **Compute J** by integrating learned p(x) around contour
5. **Verify J/ℏ ≈ n + ½** for the chosen E

## Validation Criteria

For the harmonic oscillator ground state (E = 0.5):

| Metric | Target |
|--------|--------|
| Physics loss | < 10⁻⁶ |
| J/ℏ | 0.5 ± 10⁻⁴ |
| Energy error | < 10⁻⁴ |

Repeat for n = 1, 2, 3 states to confirm the PINN approach works across multiple energy levels.

## Comparison Test

Run both Step 1 (numerical) and Step 2 (PINN) solvers on identical test cases. Results should agree to within numerical tolerance. Document:

- Training time vs numerical solve time
- Accuracy comparison
- Number of collocation points needed for convergence

## Technical Notes

- Use automatic differentiation (PyTorch/JAX) to compute dp/dx in the loss function
- The complex plane requires care: either use a complex-valued network or split into real/imaginary channels
- Contour integration can reuse the same method from Step 1, just evaluating the neural network instead of the ODE solution
- Start with a fresh network for each energy level; we'll address energy as an input parameter in Step 3

## Deliverable

A Python module with:

1. PINN architecture for learning p(x)
2. Training loop with physics loss
3. Contour integration using the learned p(x)
4. Comparison script showing agreement with Step 1 results
5. Training curves and convergence diagnostics

## Why This Matters

This step validates that neural networks can accurately represent the quantum momentum function. In higher dimensions (Step 3+), direct ODE integration fails due to the curse of dimensionality, but neural networks can potentially find sparse/compressed representations of p(x₁, x₂, ..., xₙ) that remain tractable.
