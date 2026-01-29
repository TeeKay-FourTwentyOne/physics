"""
Quantum Hamilton-Jacobi solver package.

This package implements the Leacock-Padgett formalism for finding
quantum bound state energies using contour integration in the
complex plane.

Example
-------
>>> from quantum_hj import QuantumHJSolver, HarmonicOscillator
>>>
>>> potential = HarmonicOscillator()
>>> solver = QuantumHJSolver(potential)
>>> energies = solver.find_energies(n_max=5)
>>> print(energies)  # Should be close to [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
"""

from .solver import (
    Potential,
    HarmonicOscillator,
    MorsePotential,
    QuantumHJSolver,
)
from .pinn import (
    ComplexMLP,
    MomentumPINN,
    QuantumHJPINN,
    PINNSolver,
)

__all__ = [
    'Potential',
    'HarmonicOscillator',
    'MorsePotential',
    'QuantumHJSolver',
    'ComplexMLP',
    'MomentumPINN',
    'QuantumHJPINN',
    'PINNSolver',
]
