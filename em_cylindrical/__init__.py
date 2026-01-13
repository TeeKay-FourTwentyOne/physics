"""
Einstein-Maxwell Cylindrical Symmetry Solutions

Implementation of exact solutions from:
Misra & Radhakrishna (1962) "Some Electromagnetic Fields of Cylindrical Symmetry"

This module provides symbolic and numerical calculations for the Einstein-Maxwell
field equations using the Einstein-Rosen metric with cylindrical symmetry.

Includes:
- Vacuum solutions (Cases I, II, III from the paper)
- Current density calculations and J=0 verification
- Surface currents at cylindrical boundaries
- Non-vacuum solutions with current sources
- Wave conversion module for gravitational-electromagnetic mode conversion
  (Mishima & Tomizawa 2024 method)
"""

from .solutions.case_i import CaseISolution
from .solutions.case_ii import CaseIISolution
from .solutions.case_iii import CaseIIISolution
from .metric import EinsteinRosenMetric
from .current import CurrentDensity
from .boundary import CylindricalBoundary, compute_solenoid_current
from .sources import (
    CurrentProfile,
    UniformAxialCurrent,
    SurfaceCurrent,
    CustomCurrent,
    SourcedSolution,
    uniform_axial_current,
    surface_current_cylinder,
)

# Wave conversion module (optional, for GW-EM mode conversion)
from . import wave_conversion
from .wave_conversion import (
    WaveConversion,
    ErnstPotentials,
    LinearWaveSeed,
    SolitonicSeed,
    ComplexLine,
    LagrangianPlane,
    CEnergy,
)

# Junction conditions module (Israel-Darmois formalism)
from . import junction
from .junction import (
    CylindricalHypersurface,
    ExtrinsicCurvature,
    IsraelDarmoisConditions,
    MatchedSolution,
)

__all__ = [
    # Vacuum solutions
    'CaseISolution',
    'CaseIISolution',
    'CaseIIISolution',
    'EinsteinRosenMetric',
    # Current calculations (Task 1)
    'CurrentDensity',
    # Boundary currents (Task 2)
    'CylindricalBoundary',
    'compute_solenoid_current',
    # Sourced solutions (Task 3)
    'CurrentProfile',
    'UniformAxialCurrent',
    'SurfaceCurrent',
    'CustomCurrent',
    'SourcedSolution',
    'uniform_axial_current',
    'surface_current_cylinder',
    # Wave conversion module (Mishima & Tomizawa 2024)
    'wave_conversion',
    'WaveConversion',
    'ErnstPotentials',
    'LinearWaveSeed',
    'SolitonicSeed',
    'ComplexLine',
    'LagrangianPlane',
    'CEnergy',
    # Junction conditions module (Israel-Darmois formalism)
    'junction',
    'CylindricalHypersurface',
    'ExtrinsicCurvature',
    'IsraelDarmoisConditions',
    'MatchedSolution',
]

__version__ = '0.4.0'
