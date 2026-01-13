"""
Israel-Darmois Junction Conditions Module

Implements the Israel-Darmois junction conditions for matching spacetime
regions across cylindrical hypersurfaces (ρ = R).

The junction conditions ensure that:
1. The induced metric (first fundamental form) is continuous: [γ_ab] = 0
2. The extrinsic curvature jump relates to surface stress-energy:
   [K_ab] = -8π(S_ab - ½γ_ab S)

This module provides:
- CylindricalHypersurface: Geometry of ρ = R surfaces
- ExtrinsicCurvature: Second fundamental form K_ab
- IsraelDarmoisConditions: Verification of junction conditions
- MatchedSolution: Two-region solutions with junction

Based on:
- Israel (1966), Nuovo Cimento B 44, 1
- Darmois (1927), Mémorial des Sciences Mathématiques

Usage:
    from em_cylindrical.junction import (
        MatchedSolution, IsraelDarmoisConditions
    )

    # Match interior and exterior solutions at R = 2.0
    matched = MatchedSolution(interior, exterior, junction_radius=2.0)

    # Verify junction conditions
    result = matched.junction().verify_all(t=1.0)
"""

from .hypersurface import CylindricalHypersurface
from .extrinsic_curvature import ExtrinsicCurvature
from .junction_conditions import IsraelDarmoisConditions
from .matched_solution import MatchedSolution

__all__ = [
    'CylindricalHypersurface',
    'ExtrinsicCurvature',
    'IsraelDarmoisConditions',
    'MatchedSolution',
]
