"""
Neural Network Quantum States (NNQS) Baseline Implementation.

This package implements a Carleo-Troyer style NNQS solver using
variational Monte Carlo (VMC) for comparison with LP+PINN.

Key difference from LP+PINN:
- NNQS learns: psi(x) = exp(f(x)) wavefunction
- LP+PINN learns: p(x) momentum field

Reference: Carleo & Troyer, Science 355, 602 (2017)
"""

from .networks import LogAmplitudeNetwork
from .samplers import MetropolisSampler
from .local_energy import compute_local_energy
from .train import train_nnqs, NNQSTrainer

__all__ = [
    'LogAmplitudeNetwork',
    'MetropolisSampler',
    'compute_local_energy',
    'train_nnqs',
    'NNQSTrainer',
]
