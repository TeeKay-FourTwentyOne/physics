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
    CoupledHarmonicOscillator,
    HenonHeiles,
    QuarticOscillator2D,
)
from .pinn_2d import (
    ComplexMLP2D,
    MomentumPINN2D,
    SeparableMomentumPINN2D,
    QuantumHJPINN2D,
    PINNSolver2D,
)
from .dvr import (
    DVRSolver2D,
    compute_dvr_reference,
)
from .pinn_2d_nonsep import (
    EnergyConditionedMLP,
    QuantumHJPINN2DNonSep,
    EBKQuantizer,
    NonSepPINNSolver,
)
from .pinn_2d_qn import (
    hermite_zeros,
    QuantumNumberPINN2D,
    QuantumNumberTrainer,
    NonSepQuantumSolver,
)
from .potentials_nd import (
    PotentialND,
    IsotropicOscillatorND,
    CoupledOscillator3D,
    CoupledOscillator4D,
)
from .dvr_nd import (
    DVRSolverND,
    compute_dvr_reference_nd,
)
from .pinn_nd import (
    QuantumNumberPINNND,
    QuantumNumberTrainerND,
    NonSepQuantumSolverND,
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
    # 2D separable classes
    'Potential2D',
    'HarmonicOscillator2D',
    'ComplexMLP2D',
    'MomentumPINN2D',
    'SeparableMomentumPINN2D',
    'QuantumHJPINN2D',
    'PINNSolver2D',
    # 2D non-separable potentials
    'CoupledHarmonicOscillator',
    'HenonHeiles',
    'QuarticOscillator2D',
    # DVR reference solver
    'DVRSolver2D',
    'compute_dvr_reference',
    # 2D non-separable PINN classes
    'EnergyConditionedMLP',
    'QuantumHJPINN2DNonSep',
    'EBKQuantizer',
    'NonSepPINNSolver',
    # Quantum-number-conditioned PINN classes
    'hermite_zeros',
    'QuantumNumberPINN2D',
    'QuantumNumberTrainer',
    'NonSepQuantumSolver',
    # N-dimensional potentials
    'PotentialND',
    'IsotropicOscillatorND',
    'CoupledOscillator3D',
    'CoupledOscillator4D',
    # N-dimensional DVR solver
    'DVRSolverND',
    'compute_dvr_reference_nd',
    # N-dimensional PINN solver
    'QuantumNumberPINNND',
    'QuantumNumberTrainerND',
    'NonSepQuantumSolverND',
]
