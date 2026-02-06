# Project: Leacock-Padgett Quantum Hamilton-Jacobi Solver

## Goal

Implement and validate the LP formalism for 1D quantum systems, starting with the harmonic oscillator.

## Background

The Leacock-Padgett formalism replaces the Schrödinger equation with a quantum Hamilton-Jacobi equation. For a 1D system with potential V(x), the quantum momentum function p(x) satisfies:

```
p² + iℏ(dp/dx) = 2m(E - V(x))
```

This is a Riccati equation. The key result: for bound states, the quantum action variable

```
J = (1/2πi) ∮ p(x) dx = nℏ
```

where n = 0, 1, 2, ... and the contour integral is taken in the complex x-plane around the classical turning points.

## Step 1 Task

Implement a solver for the 1D harmonic oscillator V(x) = ½mω²x² that:

1. Solves the quantum HJ equation numerically for p(x) given a trial energy E
2. Computes the contour integral J around the turning points (where E = V(x))
3. Finds energies E_n where J/ℏ equals an integer n
4. Validates against known results: E_n = (n + ½)ℏω

## Technical Notes

- The momentum p(x) is complex-valued and has square-root branch points at the classical turning points
- For the contour integral, you'll need to work in the complex x-plane, integrating around the branch cut connecting the turning points
- Use natural units where ℏ = m = ω = 1 for simplicity, so E_n should equal n + 0.5
- Start with a shooting/root-finding approach: scan E values, compute J(E), find where J/ℏ crosses integers

## Deliverable

A Python module that finds the first 5-10 energy levels of the harmonic oscillator via the LP quantization condition, with validation against exact results.
