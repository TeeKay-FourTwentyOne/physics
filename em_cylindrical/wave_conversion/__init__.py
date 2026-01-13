"""
Gravitational-Electromagnetic Wave Mode Conversion Module

This module implements the harmonic mapping method for studying nonlinear
mode conversion between gravitational and electromagnetic waves in
cylindrically symmetric spacetimes.

Based on: Mishima & Tomizawa, Phys. Rev. D 110, 024038 (2024)
arXiv:2405.04231

The module provides:
- ErnstPotentials: Complex Ernst potentials E and F
- LinearWaveSeed: Vacuum seed using cylindrical wave equation
- SolitonicSeed: Economou-Tsoubelis soliton seed
- ComplexLine: Geodesic submanifold (case a)
- LagrangianPlane: Geodesic submanifold (case b)
- CEnergy: C-energy calculations with mode decomposition
- WaveConversion: Main class combining all components

Usage:
    from em_cylindrical.wave_conversion import (
        WaveConversion, SolitonicSeed, ComplexLine
    )

    # Create solution with solitonic seed (q² - p² - l² = 1)
    seed = SolitonicSeed(p=1.0, q=1.732, l=1.0)
    submanifold = ComplexLine(theta=0.3)
    wc = WaveConversion(seed=seed, submanifold=submanifold)

    # Get mode occupancy ratios
    c = wc.c_energy()
    R_grav, R_em = c.occupancy_ratios(rho=2.0, t=1.0)
"""

from .ernst import ErnstPotentials
from .seeds import VacuumSeed, LinearWaveSeed, SolitonicSeed
from .harmonic_map import GeodesicSubmanifold, ComplexLine, LagrangianPlane
from .c_energy import CEnergy
from .conversion import WaveConversion

__all__ = [
    'ErnstPotentials',
    'VacuumSeed',
    'LinearWaveSeed',
    'SolitonicSeed',
    'GeodesicSubmanifold',
    'ComplexLine',
    'LagrangianPlane',
    'CEnergy',
    'WaveConversion',
]
