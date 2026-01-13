"""
Exact solutions of the Einstein-Maxwell field equations.

Three cases based on electromagnetic 4-potential components:
- Case (i): φ = 0 (CaseISolution)
- Case (ii): ψ = 0 (CaseIISolution)
- Case (iii): φ ≠ 0, ψ ≠ 0 (CaseIIISolution)
"""

from .case_i import CaseISolution
from .case_ii import CaseIISolution
from .case_iii import CaseIIISolution

__all__ = ['CaseISolution', 'CaseIISolution', 'CaseIIISolution']
