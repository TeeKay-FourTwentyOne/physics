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

2D Example
----------
>>> from quantum_hj import HarmonicOscillator2D, PINNSolver2D
>>>
>>> potential = HarmonicOscillator2D(omega_x=1.0, omega_y=1.0)
>>> solver = PINNSolver2D(potential)
>>> result = solver.verify_ground_state(n_epochs=5000)
>>> print(f"J_x/ℏ = {result['J_x_over_hbar']:.3f}")  # Should be ~0.5
>>> print(f"J_y/ℏ = {result['J_y_over_hbar']:.3f}")  # Should be ~0.5
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
from .potentials_2d import (
    Potential2D,
    HarmonicOscillator2D,
)
from .pinn_2d import (
    ComplexMLP2D,
    MomentumPINN2D,
    SeparableMomentumPINN2D,
    QuantumHJPINN2D,
    PINNSolver2D,
)

__all__ = [
    # 1D classes
    'Potential',
    'HarmonicOscillator',
    'MorsePotential',
    'QuantumHJSolver',
    'ComplexMLP',
    'MomentumPINN',
    'QuantumHJPINN',
    'PINNSolver',
    # 2D classes
    'Potential2D',
    'HarmonicOscillator2D',
    'ComplexMLP2D',
    'MomentumPINN2D',
    'SeparableMomentumPINN2D',
    'QuantumHJPINN2D',
    'PINNSolver2D',
]
