"""
Tests for current density, boundary currents, and sourced solutions.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from em_cylindrical import CaseISolution, CaseIISolution, CaseIIISolution
from em_cylindrical.current import CurrentDensity
from em_cylindrical.boundary import CylindricalBoundary, compute_solenoid_current
from em_cylindrical.sources import (
    UniformAxialCurrent, SurfaceCurrent, CustomCurrent,
    SourcedSolution, uniform_axial_current, surface_current_cylinder
)


class TestCurrentDensityVacuum:
    """Test that vacuum solutions have J ≈ 0."""

    def test_case_i_vacuum(self):
        """Verify J ≈ 0 for Case (i) t_only variant."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        current = CurrentDensity(sol)

        result = current.verify_vacuum(rho_val=1.5, t_val=0.5)

        # All J components should be small
        assert abs(result['J_rho']) < 0.1
        assert abs(result['J_Phi']) < 0.1
        assert abs(result['J_z']) < 0.1
        assert abs(result['J_t']) < 0.1

    def test_case_ii_vacuum(self):
        """Verify J ≈ 0 for Case (ii) rho_only variant."""
        sol = CaseIISolution(variant='rho_only', m=1.0, n=0.5, a=1.0)
        current = CurrentDensity(sol)

        result = current.verify_vacuum(rho_val=1.5, t_val=0.5)

        assert result['J_magnitude'] < 0.5  # Allow some numerical error

    def test_case_iii_vacuum(self):
        """Verify J ≈ 0 for Case (iii) variant_71."""
        sol = CaseIIISolution(variant='variant_71', m=1.0, n=0.0, a=0.5)
        current = CurrentDensity(sol)

        result = current.verify_vacuum(rho_val=1.5, t_val=0.5)

        assert result['J_magnitude'] < 0.5

    def test_vacuum_grid(self):
        """Test vacuum verification over a grid."""
        sol = CaseIISolution(variant='rho_only', m=1.0)
        current = CurrentDensity(sol)

        result = current.verify_vacuum_grid(
            rho_range=(0.5, 2.5),
            t_range=(0.0, 1.0),
            n_points=3
        )

        assert result['max_J_magnitude'] < 1.0

    def test_current_4vector_shape(self):
        """Verify current 4-vector has correct shape."""
        sol = CaseIISolution(variant='rho_only', m=1.0)
        current = CurrentDensity(sol)

        J = current.current_4vector_at(1.5, 0.5)

        assert J.shape == (4,)


class TestCylindricalBoundary:
    """Tests for surface current calculations at boundaries."""

    def test_surface_current_shape(self):
        """Verify surface current has correct shape."""
        sol = CaseIISolution(variant='rho_only', m=1.0)
        boundary = CylindricalBoundary(sol, radius=2.0)

        K = boundary.surface_current_K(t_val=0.5)

        assert K.shape == (4,)

    def test_surface_current_tangent(self):
        """Surface current should be tangent to surface (K^ρ ≈ 0)."""
        sol = CaseIISolution(variant='rho_only', m=1.0)
        boundary = CylindricalBoundary(sol, radius=2.0)

        K = boundary.surface_current_K(t_val=0.5)

        # K^ρ should be small (current tangent to surface)
        assert abs(K[0]) < 0.1

    def test_total_current_z(self):
        """Test total axial current calculation."""
        sol = CaseIISolution(variant='rho_only', m=1.0)
        boundary = CylindricalBoundary(sol, radius=2.0)

        I_z = boundary.total_current_z(t_val=0.5)

        # Should return a finite value
        assert np.isfinite(I_z)

    def test_physical_components(self):
        """Test physical interpretation of surface currents."""
        sol = CaseIISolution(variant='rho_only', m=1.0)
        boundary = CylindricalBoundary(sol, radius=2.0)

        result = boundary.surface_current_physical(t_val=0.5)

        assert 'K_z' in result
        assert 'K_Phi' in result
        assert 'K_t' in result

    def test_solenoid_current_helper(self):
        """Test solenoid current convenience function."""
        sol = CaseIISolution(variant='rho_only', m=1.0)

        result = compute_solenoid_current(sol, radius=2.0, t_val=0.5)

        assert 'total_axial_current' in result
        assert 'surface_current_density_z' in result


class TestCurrentProfiles:
    """Tests for current profile classes."""

    def test_uniform_axial_current_inside(self):
        """Test uniform current is non-zero inside cylinder."""
        current = UniformAxialCurrent(I_0=1.0, R=2.0)

        J_z = current.J_z(rho_val=1.0, t_val=0.0)

        assert J_z > 0  # Non-zero inside

    def test_uniform_axial_current_outside(self):
        """Test uniform current is zero outside cylinder."""
        current = UniformAxialCurrent(I_0=1.0, R=2.0)

        J_z = current.J_z(rho_val=3.0, t_val=0.0)

        assert J_z == 0.0  # Zero outside

    def test_uniform_axial_other_components(self):
        """Test other components are zero for axial current."""
        current = UniformAxialCurrent(I_0=1.0, R=2.0)

        assert current.J_rho(1.0, 0.0) == 0.0
        assert current.J_Phi(1.0, 0.0) == 0.0
        assert current.J_t(1.0, 0.0) == 0.0

    def test_surface_current_at_surface(self):
        """Test surface current is localized at surface."""
        current = SurfaceCurrent(K_z=1.0, R=2.0, delta=0.1)

        J_inside = current.J_z(rho_val=1.0, t_val=0.0)
        J_at_surface = current.J_z(rho_val=2.0, t_val=0.0)
        J_outside = current.J_z(rho_val=3.0, t_val=0.0)

        assert J_inside == 0.0
        assert J_at_surface > 0  # Non-zero at surface
        assert J_outside == 0.0

    def test_custom_current(self):
        """Test custom current profile."""
        custom = CustomCurrent(
            J_z_func=lambda r, t: r**2,
            J_t_func=lambda r, t: 0.5
        )

        assert custom.J_z(2.0, 0.0) == 4.0
        assert custom.J_t(1.0, 0.0) == 0.5
        assert custom.J_rho(1.0, 0.0) == 0.0

    def test_4vector_output(self):
        """Test J_4vector returns correct array."""
        current = UniformAxialCurrent(I_0=1.0, R=2.0)

        J = current.J_4vector(1.0, 0.0)

        assert J.shape == (4,)
        assert J[2] > 0  # J^z component


class TestSourcedSolution:
    """Tests for sourced solution class."""

    def test_sourced_solution_creation(self):
        """Test creation of sourced solution."""
        current = UniformAxialCurrent(I_0=1.0, R=1.0)
        sourced = SourcedSolution(current_profile=current)

        assert sourced.current is not None

    def test_field_tensor_shape(self):
        """Test field tensor has correct shape."""
        sourced = uniform_axial_current(I_0=1.0, R=1.0)

        F = sourced.field_tensor_at(0.5, 0.0)

        assert F.shape == (4, 4)

    def test_field_tensor_antisymmetric(self):
        """Test field tensor is antisymmetric."""
        sourced = uniform_axial_current(I_0=1.0, R=1.0)

        F = sourced.field_tensor_at(0.5, 0.0)

        assert np.allclose(F, -F.T)

    def test_verify_maxwell_returns_dict(self):
        """Test Maxwell verification returns expected keys."""
        sourced = uniform_axial_current(I_0=1.0, R=1.0)

        result = sourced.verify_maxwell_with_source(0.5, 0.0)

        assert 'J_z_residual' in result
        assert 'J_computed' in result
        assert 'J_specified' in result


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_uniform_axial_current_func(self):
        """Test uniform_axial_current convenience function."""
        sourced = uniform_axial_current(I_0=2.0, R=1.0)

        assert sourced is not None
        assert sourced.psi(0.5, 0.0) > 0  # Non-zero potential inside

    def test_surface_current_cylinder_func(self):
        """Test surface_current_cylinder convenience function."""
        sourced = surface_current_cylinder(K_z=1.0, R=2.0)

        assert sourced is not None
        # Inside should have zero potential
        assert abs(sourced.psi(1.0, 0.0)) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
