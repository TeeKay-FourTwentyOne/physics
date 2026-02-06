# Neural Network Approaches to High-Dimensional Quantum Ground States

## Summary for Discussion

**Authors:** Stephen + Claude (Anthropic AI assistant)  
**Date:** February 2025  
**Status:** Exploratory research, seeking expert feedback

---

## The Problem We're Interested In

Computing ground state energies for quantum systems with many degrees of freedom.

The challenge: For a quantum system with N particles (or N vibrational modes), the wavefunction ψ(x₁, x₂, ..., xₙ) lives in an N-dimensional configuration space. Traditional grid-based methods require memory that scales as M^N where M is grid points per dimension. At N=6 with M=32, this is already ~1 billion points. By N=20, it's impossible.

**Question we explored:** Can neural networks parameterize wavefunctions well enough to compute ground state energies at 50-100 dimensions?

---

## Two Approaches We Tested

### Approach 1: Physics-Informed Neural Network with Leacock-Padgett Formalism (LP+PINN)

**The idea in brief:**

The Leacock-Padgett formalism recasts the Schrödinger equation as a quantum Hamilton-Jacobi equation. Instead of solving for ψ(x) directly, we solve for p(x) = ∇S, a momentum field related to the wavefunction by ψ ∝ exp(iS/ℏ).

For bound states, this momentum field satisfies:

$$\frac{|p|^2}{2m} + V(x) - \frac{i\hbar}{2m}\nabla \cdot p = E$$

A "physics-informed neural network" (PINN) trains a neural network to output p(x) by minimizing how badly this equation is violated at randomly sampled points. The network learns a function that approximately satisfies the PDE everywhere.

**Why we thought this might work:**
- Neural networks are universal function approximators
- PINNs have succeeded on other PDEs (fluid dynamics, heat transfer)
- The momentum field p(x) might be "smoother" than ψ(x) near nodes

**What actually happened:**

| System | LP+PINN Error | Time |
|--------|---------------|------|
| 6D Coupled Oscillator | 0.45% | 20 hours |
| 6D Hénon-Heiles | 10.7% | 12 hours |

The method works but poorly. The core problem: minimizing "physics residual" (how well the PDE is satisfied) doesn't directly minimize energy error. The network can satisfy the equation reasonably well everywhere while still getting the energy wrong.

---

### Approach 2: Neural Network Quantum States (NNQS) via Variational Monte Carlo

**The idea in brief:**

This is the standard variational method, but using a neural network as the trial wavefunction.

We parameterize ψ(x) = exp(f_θ(x)) where f_θ is a neural network. The variational principle guarantees:

$$E_\theta = \frac{\langle \psi_\theta | H | \psi_\theta \rangle}{\langle \psi_\theta | \psi_\theta \rangle} \geq E_0$$

So we directly minimize E_θ with respect to network parameters θ.

**How it works in practice:**

1. Sample points {xᵢ} from |ψ_θ|² using Markov Chain Monte Carlo (MCMC)
2. At each point, compute "local energy": E_loc(x) = Hψ(x)/ψ(x)
3. The expectation ⟨E_loc⟩ over samples gives E_θ
4. Backpropagate to update θ, reducing E_θ

The key insight: if ψ_θ were the exact ground state, E_loc would equal E₀ everywhere. As the network improves, E_loc becomes constant across samples, and its variance drops to zero.

**What actually happened:**

| System | NNQS Error | Time |
|--------|------------|------|
| 6D Coupled Oscillator | 0.01% | 15 min |
| 6D Hénon-Heiles | 0.05% | 15 min |

NNQS dramatically outperforms LP+PINN: 45× more accurate in 1/80th the time.

---

## The Scaling Study

Having established NNQS works at 6D, we tested scaling to higher dimensions.

**Test system:** Hénon-Heiles potential with all-to-all coupling

$$V(x) = \frac{1}{2}\sum_i x_i^2 + \frac{\lambda}{N}\sum_{i<j}\left(x_i^2 x_j - \frac{x_j^3}{3}\right)$$

The 1/N normalization keeps the anharmonic contribution O(1) as N grows.

**Why all-to-all coupling?**

The standard benchmark uses nearest-neighbor (chain) coupling. Tensor network methods like ML-MCTDH exploit this chain structure brilliantly—they've solved 1458D chain-coupled systems. But all-to-all coupling breaks the tree-tensor decomposition that makes ML-MCTDH efficient. This is a regime where neural approaches might have an advantage.

**Validation strategy:**

Above ~20D, no method gives exact reference energies. We validated by:
1. Anchoring at 6D against known Nest & Meyer (2002) benchmark
2. Self-consistency: 3 independent training runs must agree within 1%
3. Extensive scaling: E₀/N should approach 0.5 (the harmonic limit dominates)

---

## Results

| Dimension | Energy (E/N) | Self-consistency (CV) | Status |
|-----------|--------------|----------------------|--------|
| 6D | 0.500 | 0.006% | ✓ Pass |
| 12D | 0.500 | 0.01% | ✓ Pass |
| 24D | 0.502 | 0.13% | ✓ Pass |
| 50D | 0.508 | 0.14% | ✓ Pass |
| 100D | — | 8.7% | ✗ Fail |

**The 100D failure mode:** MCMC acceptance rate collapsed to 1%. 

Standard Metropolis-Hastings proposes random displacements; in 100D, random moves are almost always "uphill" in energy and get rejected. The walkers froze, the network fit those frozen positions, and fresh validation revealed it learned nothing generalizable.

**Current status:** We're implementing Hamiltonian Monte Carlo (HMC), which uses gradient information to make proposals that follow the probability landscape. HMC acceptance should remain high regardless of dimension. Testing in progress.

---

## Summary of Findings

1. **LP+PINN doesn't work well for ground state energies.** Minimizing physics residual ≠ minimizing energy error. The 10.7% error on Hénon-Heiles is unacceptable.

2. **NNQS (VMC with neural wavefunctions) works excellently through 50D.** Self-consistent to 0.14% coefficient of variation, matching expected extensive scaling.

3. **Basic MCMC fails at 100D.** This is a known limitation of random-walk samplers in high dimensions, not a failure of neural wavefunctions per se.

4. **All-to-all coupling may be a good test case** for neural methods since it breaks the structure that tensor networks exploit.

---

## Questions for Discussion

1. **Is all-to-all Hénon-Heiles actually hard for ML-MCTDH?** Or can it be reformulated to restore tree structure?

2. **Do you know of any benchmarks in the 20-100D range?** Even approximate references would help validate our results.

3. **Is the E/N ≈ 0.5 result physically reasonable?** For weak all-to-all coupling (λ/N ≈ 0.001), should the energy be essentially harmonic?

4. **Is there a real application where this matters?** We can compute ground state energies for 50D all-to-all coupled anharmonic systems in ~2 hours on a laptop. Is this useful for anything?

5. **What validation standard would satisfy a referee?** Self-consistency + extensive scaling + 6D anchor is our current approach. Is this sufficient?

---

## Technical Details (For Reference)

**Network architecture:** 4-layer MLP, 512 hidden units, tanh activation, ~1M parameters

**Wavefunction ansatz:** ψ(x) = exp(f_θ(x)) where f_θ: ℝᴺ → ℝ

**Local energy computation:** 
$$E_{loc}(x) = -\frac{\hbar^2}{2m}\left(\nabla^2 f + |\nabla f|^2\right) + V(x)$$

**MCMC:** Metropolis-Hastings with adaptive step size targeting 30-50% acceptance

**Training:** Adam optimizer, learning rate 10⁻³ with decay, ~10,000 epochs

**Compute:** Local laptop with GPU, ~15-20 min for 6D, ~4 hours for 50D

---

## Code Availability

All code developed during this exploration is available for review. Implementation is in PyTorch.
