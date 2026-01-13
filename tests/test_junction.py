"""
Tests for the junction conditions module.

Tests the Israel-Darmois junction conditions implementation for
matching spacetime regions across cylindrical hypersurfaces.
"""

import pytest
import numpy as np

from em_cylindrical import CaseISolution, CaseIISolution
from em_cylindrical.junction import (
    CylindricalHypersurface,
    ExtrinsicCurvature,
    IsraelDarmoisConditions,
    MatchedSolution,
)


class TestCylindricalHypersurface:
    """Tests for CylindricalHypersurface class."""

    def test_hypersurface_creation(self):
        """Test basic hypersurface creation."""
        hs = CylindricalHypersurface(radius=2.0)
        assert hs.radius == 2.0
        assert hs.R == 2.0

    def test_hypersurface_invalid_radius(self):
        """Test that invalid radius raises error."""
        with pytest.raises(ValueError):
            CylindricalHypersurface(radius=0.0)
        with pytest.raises(ValueError):
            CylindricalHypersurface(radius=-1.0)

    def test_tangent_coordinates(self):
        """Test tangent coordinate names."""
        hs = CylindricalHypersurface(radius=1.0)
        coords = hs.tangent_coordinates()
        assert coords == ['Phi', 'z', 't']

    def test_induced_metric_shape(self):
        """Test induced metric returns 3x3 matrix."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        hs = CylindricalHypersurface(radius=2.0)

        gamma = hs.induced_metric(sol, t=1.0)
        assert gamma.shape == (3, 3)

    def test_induced_metric_diagonal(self):
        """Test induced metric is diagonal."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        hs = CylindricalHypersurface(radius=2.0)

        gamma = hs.induced_metric(sol, t=1.0)

        # Check off-diagonal elements are zero
        assert gamma[0, 1] == 0.0
        assert gamma[0, 2] == 0.0
        assert gamma[1, 0] == 0.0
        assert gamma[1, 2] == 0.0
        assert gamma[2, 0] == 0.0
        assert gamma[2, 1] == 0.0

    def test_induced_metric_inverse(self):
        """Test induced metric inverse is correct."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        hs = CylindricalHypersurface(radius=2.0)

        gamma = hs.induced_metric(sol, t=1.0)
        gamma_inv = hs.induced_metric_inverse(sol, t=1.0)

        # gamma @ gamma_inv should be identity
        product = gamma @ gamma_inv
        identity = np.eye(3)
        np.testing.assert_allclose(product, identity, atol=1e-10)

    def test_induced_metric_determinant(self):
        """Test induced metric determinant."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        hs = CylindricalHypersurface(radius=2.0)

        det = hs.induced_metric_determinant(sol, t=1.0)
        gamma = hs.induced_metric(sol, t=1.0)

        # Compare with numpy determinant
        np.testing.assert_allclose(det, np.linalg.det(gamma), rtol=1e-10)

    def test_normal_vector_orthogonal(self):
        """Test normal vector is purely radial."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        hs = CylindricalHypersurface(radius=2.0)

        n = hs.normal_vector(sol, t=1.0)

        # Only radial component should be non-zero
        assert n[0] != 0.0  # n^rho
        assert n[1] == 0.0  # n^Phi
        assert n[2] == 0.0  # n^z
        assert n[3] == 0.0  # n^t

    def test_normal_covector_outward(self):
        """Test normal covector points outward."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        hs = CylindricalHypersurface(radius=2.0)

        n = hs.normal_covector(sol, t=1.0)

        # n_rho should be positive (outward)
        assert n[0] > 0.0

    def test_area_element_positive(self):
        """Test area element is positive."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        hs = CylindricalHypersurface(radius=2.0)

        area = hs.area_element(sol, t=1.0)
        assert area > 0


class TestExtrinsicCurvature:
    """Tests for ExtrinsicCurvature class."""

    def test_extrinsic_curvature_creation(self):
        """Test ExtrinsicCurvature creation."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        hs = CylindricalHypersurface(radius=2.0)
        K = ExtrinsicCurvature(hs, sol)

        assert K.R == 2.0

    def test_K_ab_shape(self):
        """Test K_ab returns 3x3 matrix."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        hs = CylindricalHypersurface(radius=2.0)
        K = ExtrinsicCurvature(hs, sol)

        K_ab = K.K_ab(t=1.0)
        assert K_ab.shape == (3, 3)

    def test_K_ab_diagonal(self):
        """Test K_ab is diagonal for diagonal metric."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        hs = CylindricalHypersurface(radius=2.0)
        K = ExtrinsicCurvature(hs, sol)

        K_ab = K.K_ab(t=1.0)

        # Check off-diagonal elements are zero
        assert K_ab[0, 1] == 0.0
        assert K_ab[0, 2] == 0.0
        assert K_ab[1, 0] == 0.0
        assert K_ab[1, 2] == 0.0
        assert K_ab[2, 0] == 0.0
        assert K_ab[2, 1] == 0.0

    def test_K_trace_finite(self):
        """Test K trace is finite."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        hs = CylindricalHypersurface(radius=2.0)
        K = ExtrinsicCurvature(hs, sol)

        trace = K.K_trace(t=1.0)
        assert np.isfinite(trace)

    def test_K_squared_non_negative(self):
        """Test K_ab K^ab is non-negative."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        hs = CylindricalHypersurface(radius=2.0)
        K = ExtrinsicCurvature(hs, sol)

        K_sq = K.K_squared(t=1.0)
        assert K_sq >= 0

    def test_principal_curvatures(self):
        """Test principal curvatures are real."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        hs = CylindricalHypersurface(radius=2.0)
        K = ExtrinsicCurvature(hs, sol)

        principal = K.principal_curvatures(t=1.0)
        assert len(principal) == 3
        assert all(np.isreal(p) for p in principal)

    def test_summary_keys(self):
        """Test summary contains expected keys."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        hs = CylindricalHypersurface(radius=2.0)
        K = ExtrinsicCurvature(hs, sol)

        summary = K.summary(t=1.0)
        expected_keys = ['K_PhiPhi', 'K_zz', 'K_tt', 'K_trace', 'K_squared']
        for key in expected_keys:
            assert key in summary


class TestIsraelDarmoisConditions:
    """Tests for IsraelDarmoisConditions class."""

    def test_junction_creation(self):
        """Test IsraelDarmoisConditions creation."""
        sol_int = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        sol_ext = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

        jc = IsraelDarmoisConditions(sol_int, sol_ext, radius=2.0)
        assert jc.radius == 2.0

    def test_same_solution_first_condition(self):
        """Test first condition satisfied for identical solutions."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

        jc = IsraelDarmoisConditions(sol, sol, radius=2.0)
        result = jc.first_condition(t=1.0)

        assert result['satisfied']
        assert result['max_jump'] < 1e-10

    def test_same_solution_second_condition(self):
        """Test second condition for identical solutions (no thin shell)."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

        jc = IsraelDarmoisConditions(sol, sol, radius=2.0)
        result = jc.second_condition(t=1.0)

        # Same solution should give no thin shell
        assert not result['thin_shell']

    def test_different_solutions_first_condition(self):
        """Test first condition may fail for different solutions."""
        sol_int = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        sol_ext = CaseISolution(alpha=2.0, beta=0.8, variant='t_only', m=2.0)

        jc = IsraelDarmoisConditions(sol_int, sol_ext, radius=2.0)
        result = jc.first_condition(t=1.0)

        # Different solutions likely have metric discontinuity
        assert result['max_jump'] > 0

    def test_surface_stress_energy_shape(self):
        """Test surface stress-energy tensor has correct shape."""
        sol_int = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        sol_ext = CaseISolution(alpha=2.0, beta=0.8, variant='t_only', m=2.0)

        jc = IsraelDarmoisConditions(sol_int, sol_ext, radius=2.0)
        S = jc.surface_stress_energy(t=1.0)

        assert S.shape == (3, 3)

    def test_verify_all_keys(self):
        """Test verify_all returns expected keys."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

        jc = IsraelDarmoisConditions(sol, sol, radius=2.0)
        result = jc.verify_all(t=1.0)

        assert 'time' in result
        assert 'radius' in result
        assert 'first_condition' in result
        assert 'second_condition' in result
        assert 'valid_junction' in result

    def test_summary_output(self):
        """Test summary produces string output."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

        jc = IsraelDarmoisConditions(sol, sol, radius=2.0)
        summary = jc.summary(t=1.0)

        assert isinstance(summary, str)
        assert 'Israel-Darmois' in summary


class TestMatchedSolution:
    """Tests for MatchedSolution class."""

    def test_matched_solution_creation(self):
        """Test MatchedSolution creation."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

        matched = MatchedSolution(sol, sol, junction_radius=2.0, validate=False)
        assert matched.R == 2.0

    def test_region_selection(self):
        """Test correct region is selected."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

        matched = MatchedSolution(sol, sol, junction_radius=2.0, validate=False)

        assert matched.region(1.0) == 'interior'
        assert matched.region(2.0) == 'exterior'
        assert matched.region(3.0) == 'exterior'

    def test_is_at_junction(self):
        """Test junction detection."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

        matched = MatchedSolution(sol, sol, junction_radius=2.0, validate=False)

        assert matched.is_at_junction(2.0)
        assert not matched.is_at_junction(1.5)
        assert not matched.is_at_junction(2.5)

    def test_mu_continuous(self):
        """Test mu is evaluated in both regions."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

        matched = MatchedSolution(sol, sol, junction_radius=2.0, validate=False)

        mu_int = matched.mu(1.5, 1.0)
        mu_ext = matched.mu(2.5, 1.0)

        assert np.isfinite(mu_int)
        assert np.isfinite(mu_ext)

    def test_junction_method(self):
        """Test junction() returns IsraelDarmoisConditions."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

        matched = MatchedSolution(sol, sol, junction_radius=2.0, validate=False)

        jc = matched.junction()
        assert isinstance(jc, IsraelDarmoisConditions)

    def test_is_valid(self):
        """Test is_valid for matched solutions."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

        matched = MatchedSolution(sol, sol, junction_radius=2.0, validate=False)

        # Same solution should be valid
        assert matched.is_valid(t=1.0)

    def test_has_thin_shell(self):
        """Test thin shell detection."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

        matched = MatchedSolution(sol, sol, junction_radius=2.0, validate=False)

        # Same solution should have no thin shell
        assert not matched.has_thin_shell(t=1.0)

    def test_evaluate_profile(self):
        """Test radial profile evaluation."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

        matched = MatchedSolution(sol, sol, junction_radius=2.0, validate=False)

        profile = matched.evaluate_profile(t=1.0, rho_min=0.5, rho_max=4.0, n_points=20)

        assert 'rho' in profile
        assert 'mu' in profile
        assert 'lambda' in profile
        assert len(profile['rho']) == 20

    def test_summary_output(self):
        """Test summary produces string output."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

        matched = MatchedSolution(sol, sol, junction_radius=2.0, validate=False)

        summary = matched.summary(t=1.0)
        assert isinstance(summary, str)
        assert 'Matched Solution' in summary


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow_same_solution(self):
        """Test complete workflow with identical interior/exterior."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

        # Create matched solution
        matched = MatchedSolution(sol, sol, junction_radius=2.0, validate=False)

        # Check junction conditions at multiple times
        for t in [0.0, 0.5, 1.0]:
            result = matched.junction().verify_all(t)

            # First condition should be satisfied
            assert result['first_condition']['satisfied']

            # No thin shell for same solution
            assert not result['second_condition']['thin_shell']

    def test_full_workflow_case_ii(self):
        """Test workflow with Case II solution."""
        sol = CaseIISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

        matched = MatchedSolution(sol, sol, junction_radius=2.0, validate=False)

        # Verify at t=1.0
        result = matched.junction().verify_all(t=1.0)
        assert result['first_condition']['satisfied']

    def test_extrinsic_curvature_consistency(self):
        """Test extrinsic curvature computed two ways gives same result."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        hs = CylindricalHypersurface(radius=2.0)

        K = ExtrinsicCurvature(hs, sol)

        # Get K_ab and compute trace two ways
        K_ab = K.K_ab(t=1.0)
        trace_direct = K.K_trace(t=1.0)

        gamma_inv = hs.induced_metric_inverse(sol, t=1.0)
        trace_manual = np.sum(gamma_inv * K_ab)

        np.testing.assert_allclose(trace_direct, trace_manual, rtol=1e-10)

    def test_different_solutions_thin_shell(self):
        """Test different solutions may produce thin shell."""
        sol_int = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        sol_ext = CaseISolution(alpha=1.5, beta=0.7, variant='t_only', m=1.5)

        jc = IsraelDarmoisConditions(sol_int, sol_ext, radius=2.0)

        result = jc.second_condition(t=1.0)

        # Different solutions typically have K_ab jump
        # (whether this implies thin_shell depends on tolerance)
        assert 'K_jump' in result


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_radius(self):
        """Test behavior at small radius."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        hs = CylindricalHypersurface(radius=0.1)

        # Should still compute induced metric
        gamma = hs.induced_metric(sol, t=1.0)
        assert all(np.isfinite(gamma.flatten()))

    def test_large_radius(self):
        """Test behavior at moderately large radius."""
        # Use moderate radius to avoid overflow in metric exponentials
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        hs = CylindricalHypersurface(radius=10.0)

        gamma = hs.induced_metric(sol, t=1.0)
        assert all(np.isfinite(gamma.flatten()))

    def test_zero_time(self):
        """Test evaluation at t=0."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        hs = CylindricalHypersurface(radius=2.0)

        gamma = hs.induced_metric(sol, t=0.0)
        assert all(np.isfinite(gamma.flatten()))

    def test_negative_time(self):
        """Test evaluation at negative time."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        hs = CylindricalHypersurface(radius=2.0)

        gamma = hs.induced_metric(sol, t=-1.0)
        assert all(np.isfinite(gamma.flatten()))
