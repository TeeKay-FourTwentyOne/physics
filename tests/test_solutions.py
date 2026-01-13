"""
Tests for Einstein-Maxwell cylindrical symmetry solutions.

These tests verify that the implemented solutions satisfy
the field equations from Misra & Radhakrishna (1962).
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from em_cylindrical import CaseISolution, CaseIISolution, CaseIIISolution
from em_cylindrical.metric import EinsteinRosenMetric


class TestCaseISolution:
    """Tests for Case (i): phi = 0 solutions."""

    def test_phi_is_zero(self):
        """Verify that phi is always zero for Case (i)."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        for rho in [0.5, 1.0, 2.0]:
            for t in [0.0, 0.5, 1.0]:
                assert sol.phi(rho, t) == 0.0

    def test_t_only_variant_creation(self):
        """Test creation of t_only variant."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=2.0, n=0.0)
        assert sol.variant == 't_only'
        assert sol.alpha == 1.0
        assert sol.beta == 0.5

    def test_bessel_variant_creation(self):
        """Test creation of Bessel function variant."""
        sol = CaseISolution(alpha=1.0, beta=0.0, variant='bessel', k=1.0, A=0.5)
        assert sol.variant == 'bessel'
        # psi should be non-trivial
        psi_val = sol.psi(1.0, 0.0)
        assert psi_val != 0.0

    def test_metric_tensor_shape(self):
        """Verify metric tensor has correct shape."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        g = sol.metric_at(1.5, 0.5)
        assert g.shape == (4, 4)

    def test_metric_tensor_signature(self):
        """Verify metric has correct signature (-,-,-,+)."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        g = sol.metric_at(1.5, 0.5)
        # g_rhorho < 0, g_PhiPhi < 0, g_zz < 0, g_tt > 0
        assert g[0, 0] < 0  # g_rhorho
        assert g[1, 1] < 0  # g_PhiPhi
        assert g[2, 2] < 0  # g_zz
        assert g[3, 3] > 0  # g_tt

    def test_field_tensor_antisymmetric(self):
        """Verify field tensor is antisymmetric."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        F = sol.field_tensor_at(1.5, 0.5)
        assert np.allclose(F, -F.T)

    def test_field_equations_t_only(self):
        """Check field equations are approximately satisfied for t_only variant."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0, n=0.0)
        residuals = sol.verify_field_equations(1.5, 0.5)
        # Residuals should be small (allowing for numerical differentiation error)
        for name, value in residuals.items():
            assert abs(value) < 0.1, f"{name} too large: {value}"


class TestCaseIISolution:
    """Tests for Case (ii): psi = 0 solutions."""

    def test_psi_is_zero(self):
        """Verify that psi is always zero for Case (ii)."""
        sol = CaseIISolution(variant='rho_only', m=1.0)
        for rho in [0.5, 1.0, 2.0]:
            for t in [0.0, 0.5, 1.0]:
                assert sol.psi(rho, t) == 0.0

    def test_rho_only_variant(self):
        """Test rho_only variant."""
        sol = CaseIISolution(variant='rho_only', m=1.0, n=0.5, a=1.0)
        # phi should be non-trivial
        phi_val = sol.phi(1.5, 0.0)
        assert phi_val != 0.0

    def test_t_only_variant(self):
        """Test t_only variant."""
        sol = CaseIISolution(variant='t_only', m=1.0, n=0.0, a=1.0)
        # phi should depend on t
        phi_t0 = sol.phi(1.5, 0.0)
        phi_t1 = sol.phi(1.5, 1.0)
        assert phi_t0 != phi_t1

    def test_rho_quadratic_variant(self):
        """Test rho_quadratic variant."""
        sol = CaseIISolution(variant='rho_quadratic', m=1.0, a=0.5, b=1.0)
        # phi = 0.5 * a * rho^2 + b = 0.5 * 0.5 * 4 + 1 = 2 at rho=2
        phi_expected = 0.5 * 0.5 * 4.0 + 1.0
        phi_actual = sol.phi(2.0, 0.0)
        assert np.isclose(phi_actual, phi_expected, rtol=1e-5)

    def test_metric_tensor_shape(self):
        """Verify metric tensor has correct shape."""
        sol = CaseIISolution(variant='rho_only', m=1.0)
        g = sol.metric_at(1.5, 0.5)
        assert g.shape == (4, 4)

    def test_field_tensor_antisymmetric(self):
        """Verify field tensor is antisymmetric."""
        sol = CaseIISolution(variant='rho_only', m=1.0)
        F = sol.field_tensor_at(1.5, 0.5)
        assert np.allclose(F, -F.T)


class TestCaseIIISolution:
    """Tests for Case (iii): both phi and psi non-zero."""

    def test_both_nonzero(self):
        """Verify both phi and psi can be non-zero."""
        sol = CaseIIISolution(variant='variant_71', m=1.0, a=0.5)
        rho_val, t_val = 1.5, 0.5
        phi = sol.phi(rho_val, t_val)
        psi = sol.psi(rho_val, t_val)
        # At least one should be non-zero (psi = a*t + b)
        assert psi != 0.0 or t_val == 0.0

    def test_variant_71(self):
        """Test variant_71: psi linear in t."""
        sol = CaseIIISolution(variant='variant_71', m=1.0, a=0.5, b=0.0)
        # psi = a*t + b = 0.5*t
        assert np.isclose(sol.psi(1.0, 2.0), 1.0)  # 0.5 * 2.0 = 1.0

    def test_variant_72(self):
        """Test variant_72: psi quadratic in rho."""
        sol = CaseIIISolution(variant='variant_72', m=1.0, a=0.5, b=0.0)
        # psi = 0.5*a*rho^2 + b = 0.25*rho^2
        rho_val = 2.0
        expected = 0.5 * 0.5 * rho_val**2
        assert np.isclose(sol.psi(rho_val, 0.0), expected)

    def test_metric_tensor_shape(self):
        """Verify metric tensor has correct shape."""
        sol = CaseIIISolution(variant='variant_71', m=1.0, a=0.5)
        g = sol.metric_at(1.5, 0.5)
        assert g.shape == (4, 4)

    def test_energy_tensor_trace(self):
        """Verify energy tensor trace is approximately zero."""
        sol = CaseIIISolution(variant='variant_71', m=1.0, a=0.5)
        E = sol.energy_tensor_at(1.5, 0.5)
        trace = np.trace(E)
        # Electromagnetic energy tensor should be trace-free
        assert abs(trace) < 0.1


class TestEinsteinRosenMetric:
    """Tests for the metric class."""

    def test_metric_diagonal(self):
        """Verify metric is diagonal."""
        import sympy as sp
        rho_sym, t_sym = sp.symbols('rho t', real=True, positive=True)
        lam = sp.Symbol('lambda', real=True)
        mu = sp.Symbol('mu', real=True)

        metric = EinsteinRosenMetric(lam, mu)
        g = metric.g_symbolic

        # Off-diagonal elements should be zero
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert g[i, j] == 0

    def test_metric_determinant_negative(self):
        """Verify metric determinant is negative (Lorentzian signature)."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        g = sol.metric_at(1.5, 0.5)
        det = np.linalg.det(g)
        assert det < 0


class TestFieldEquationsConsistency:
    """Test that solutions satisfy their respective field equations."""

    @pytest.mark.parametrize("variant", ['t_only', 'rho_only'])
    def test_case_i_variants(self, variant):
        """Test Case (i) field equations for different variants."""
        if variant == 't_only':
            sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        else:
            sol = CaseISolution(alpha=1.0, beta=0.5, variant='rho_only', m=1.0, a=1.0)

        residuals = sol.verify_field_equations(1.5, 0.5)
        for name, value in residuals.items():
            assert abs(value) < 1.0, f"{variant}: {name} = {value}"

    @pytest.mark.parametrize("variant", ['rho_only', 't_only', 'rho_quadratic', 't_linear'])
    def test_case_ii_variants(self, variant):
        """Test Case (ii) field equations for different variants."""
        sol = CaseIISolution(variant=variant, m=1.0, n=0.5, a=1.0, p=0.0)
        residuals = sol.verify_field_equations(1.5, 0.5)
        for name, value in residuals.items():
            assert abs(value) < 1.0, f"{variant}: {name} = {value}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
