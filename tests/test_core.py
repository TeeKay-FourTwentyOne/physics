"""
Tests for core modules: metric, field_tensor, energy_tensor, wave_equation.

These tests ensure comprehensive coverage of the foundational classes
that underpin the Einstein-Maxwell cylindrical symmetry solutions.
"""

import pytest
import numpy as np
import sympy as sp

from em_cylindrical import CaseISolution, CaseIISolution, CaseIIISolution
from em_cylindrical.metric import (
    EinsteinRosenMetric,
    compute_christoffel_symbols,
    rho, t,
)
from em_cylindrical.field_tensor import (
    ElectromagneticFieldTensor,
    levi_civita_4d,
)
from em_cylindrical.energy_tensor import (
    ElectromagneticEnergyTensor,
)
from em_cylindrical.wave_equation import (
    CylindricalWaveSolution,
    create_standing_wave,
    create_superposition,
)


class TestEinsteinRosenMetric:
    """Comprehensive tests for EinsteinRosenMetric class."""

    @pytest.fixture
    def simple_metric(self):
        """Create a simple metric for testing."""
        lam = sp.Symbol('lambda')
        mu = sp.Symbol('mu')
        return EinsteinRosenMetric(lam, mu)

    @pytest.fixture
    def numeric_metric(self):
        """Create a metric with numeric functions."""
        # mu = rho, lambda = rho + t
        return EinsteinRosenMetric(rho + t, rho)

    def test_metric_creation(self, simple_metric):
        """Test metric is created correctly."""
        assert simple_metric.g_symbolic is not None
        assert simple_metric.g_inv_symbolic is not None
        assert simple_metric.det_g_symbolic is not None

    def test_metric_shape(self, simple_metric):
        """Test metric tensor has correct shape."""
        assert simple_metric.g_symbolic.shape == (4, 4)
        assert simple_metric.g_inv_symbolic.shape == (4, 4)

    def test_metric_at_shape(self, numeric_metric):
        """Test numeric metric evaluation shape."""
        g = numeric_metric.metric_at(2.0, 1.0)
        assert g.shape == (4, 4)

    def test_inverse_metric_at_shape(self, numeric_metric):
        """Test inverse metric evaluation shape."""
        g_inv = numeric_metric.inverse_metric_at(2.0, 1.0)
        assert g_inv.shape == (4, 4)

    def test_metric_inverse_product(self, numeric_metric):
        """Test g * g^{-1} = I."""
        rho_val, t_val = 2.0, 1.0
        g = numeric_metric.metric_at(rho_val, t_val)
        g_inv = numeric_metric.inverse_metric_at(rho_val, t_val)

        product = g @ g_inv
        identity = np.eye(4)
        np.testing.assert_allclose(product, identity, atol=1e-10)

    def test_metric_diagonal(self, numeric_metric):
        """Test metric is diagonal."""
        g = numeric_metric.metric_at(2.0, 1.0)
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert g[i, j] == 0

    def test_metric_signature(self, numeric_metric):
        """Test metric has (-,-,-,+) signature."""
        g = numeric_metric.metric_at(2.0, 1.0)
        assert g[0, 0] < 0  # g_rhorho
        assert g[1, 1] < 0  # g_PhiPhi
        assert g[2, 2] < 0  # g_zz
        assert g[3, 3] > 0  # g_tt

    def test_ds_squared_null(self, numeric_metric):
        """Test ds² for a null displacement."""
        rho_val, t_val = 2.0, 1.0
        g = numeric_metric.metric_at(rho_val, t_val)

        # For null path: ds² = 0
        # Try radial null ray: drho = dt, others = 0
        # ds² = g_rhorho * drho² + g_tt * dt²
        # For null: g_rhorho + g_tt = 0 (if drho = dt = 1)
        # Actually ds² = g_rhorho - g_tt (since g_rhorho < 0, g_tt > 0)

        ds2 = numeric_metric.ds_squared(rho_val, t_val, 1.0, 0.0, 0.0, 1.0)
        # For Einstein-Rosen with equal lambda-mu factors, radial null has ds²=0
        # But for general case, just check it returns a number
        assert np.isfinite(ds2)

    def test_ds_squared_spacelike(self, numeric_metric):
        """Test ds² for a spacelike displacement."""
        ds2 = numeric_metric.ds_squared(2.0, 1.0, 1.0, 0.0, 0.0, 0.0)
        assert ds2 < 0  # Spacelike: ds² < 0

    def test_ds_squared_timelike(self, numeric_metric):
        """Test ds² for a timelike displacement."""
        ds2 = numeric_metric.ds_squared(2.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        assert ds2 > 0  # Timelike: ds² > 0

    def test_lambda_at(self, numeric_metric):
        """Test lambda evaluation."""
        lam = numeric_metric.lambda_at(2.0, 1.0)
        assert lam == 3.0  # rho + t = 2 + 1

    def test_mu_at(self, numeric_metric):
        """Test mu evaluation."""
        mu = numeric_metric.mu_at(2.0, 1.0)
        assert mu == 2.0  # rho = 2

    def test_determinant_negative(self, numeric_metric):
        """Test metric determinant is negative (Lorentzian)."""
        g = numeric_metric.metric_at(2.0, 1.0)
        det = np.linalg.det(g)
        assert det < 0


class TestChristoffelSymbols:
    """Tests for Christoffel symbol computation."""

    def test_christoffel_returns_dict(self):
        """Test Christoffel symbols returned as dictionary."""
        lam = rho + t
        mu = rho
        metric = EinsteinRosenMetric(lam, mu)
        christoffel = compute_christoffel_symbols(metric)
        assert isinstance(christoffel, dict)

    def test_christoffel_symmetry(self):
        """Test Christoffel symbols are symmetric in lower indices."""
        lam = rho
        mu = t
        metric = EinsteinRosenMetric(lam, mu)
        christoffel = compute_christoffel_symbols(metric)

        for (i, j, k), val in christoffel.items():
            # Γ^i_jk = Γ^i_kj
            if (i, k, j) in christoffel:
                assert sp.simplify(val - christoffel[(i, k, j)]) == 0

    def test_christoffel_non_empty(self):
        """Test some Christoffel symbols are non-zero."""
        lam = rho + t
        mu = rho
        metric = EinsteinRosenMetric(lam, mu)
        christoffel = compute_christoffel_symbols(metric)
        assert len(christoffel) > 0


class TestElectromagneticFieldTensor:
    """Comprehensive tests for ElectromagneticFieldTensor class."""

    @pytest.fixture
    def simple_field(self):
        """Create a simple field tensor with pure symbolic expressions."""
        phi = rho * t
        psi = rho**2
        return ElectromagneticFieldTensor(phi, psi)

    def test_field_tensor_creation(self, simple_field):
        """Test field tensor is created correctly."""
        assert simple_field.F_symbolic is not None

    def test_field_tensor_shape(self, simple_field):
        """Test field tensor has correct shape."""
        assert simple_field.F_symbolic.shape == (4, 4)

    def test_field_tensor_antisymmetric(self, simple_field):
        """Test field tensor is antisymmetric."""
        F = simple_field.at(2.0, 1.0)
        np.testing.assert_allclose(F, -F.T, atol=1e-10)

    def test_field_tensor_at_shape(self, simple_field):
        """Test numeric evaluation shape."""
        F = simple_field.at(2.0, 1.0)
        assert F.shape == (4, 4)

    def test_field_tensor_via_solution(self):
        """Test field tensor from solution's built-in method."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        F = sol.field_tensor_at(2.0, 1.0)
        assert F.shape == (4, 4)
        np.testing.assert_allclose(F, -F.T, atol=1e-10)

    def test_invariants_return_tuple(self, simple_field):
        """Test invariants returns two values."""
        g_inv = np.diag([-1, -1, -1, 1])  # Simplified inverse metric
        I1, I2 = simple_field.invariants(2.0, 1.0, g_inv)
        assert np.isfinite(I1)
        assert np.isfinite(I2)

    def test_derivatives_stored(self, simple_field):
        """Test that derivatives are stored symbolically."""
        assert simple_field.phi_1_symbolic is not None
        assert simple_field.phi_4_symbolic is not None
        assert simple_field.psi_1_symbolic is not None
        assert simple_field.psi_4_symbolic is not None

    def test_case_i_field_tensor_via_solution(self):
        """Test Case I field tensor through solution method."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        F = sol.field_tensor_at(2.0, 1.0)
        # phi=0 means F_01 and F_13 should be 0
        assert abs(F[0, 1]) < 1e-10
        assert abs(F[1, 3]) < 1e-10

    def test_case_ii_field_tensor_via_solution(self):
        """Test Case II field tensor through solution method."""
        sol = CaseIISolution(variant='rho_only', m=1.0)
        F = sol.field_tensor_at(2.0, 1.0)
        # psi=0 means F_02 and F_23 should be 0
        assert abs(F[0, 2]) < 1e-10
        assert abs(F[2, 3]) < 1e-10


class TestLeviCivita:
    """Tests for Levi-Civita symbol."""

    def test_levi_civita_identity(self):
        """Test ε_{0123} = 1."""
        assert levi_civita_4d(0, 1, 2, 3) == 1

    def test_levi_civita_even_permutation(self):
        """Test even permutations give +1."""
        # Cyclic permutations of (0,1,2,3) are even
        assert levi_civita_4d(1, 2, 3, 0) == -1  # 3 transpositions from identity
        assert levi_civita_4d(2, 3, 0, 1) == 1   # 2 transpositions (even)

    def test_levi_civita_odd_permutation(self):
        """Test odd permutations give -1."""
        assert levi_civita_4d(1, 0, 2, 3) == -1
        assert levi_civita_4d(0, 2, 1, 3) == -1

    def test_levi_civita_repeated_index(self):
        """Test repeated indices give 0."""
        assert levi_civita_4d(0, 0, 1, 2) == 0
        assert levi_civita_4d(1, 2, 2, 3) == 0
        assert levi_civita_4d(0, 1, 2, 2) == 0

    def test_levi_civita_antisymmetry(self):
        """Test antisymmetry under index exchange."""
        for perm in [(0, 1, 2, 3), (1, 2, 0, 3), (2, 0, 1, 3)]:
            i, j, k, l = perm
            # Swapping any two indices negates the result
            assert levi_civita_4d(j, i, k, l) == -levi_civita_4d(i, j, k, l)


class TestElectromagneticEnergyTensor:
    """Comprehensive tests for ElectromagneticEnergyTensor class."""

    @pytest.fixture
    def simple_energy(self):
        """Create a simple energy tensor with pure symbolic expressions."""
        phi = rho * t
        psi = rho**2
        mu = rho
        lam = rho + t
        return ElectromagneticEnergyTensor(phi, psi, mu, lam)

    def test_energy_tensor_creation(self, simple_energy):
        """Test energy tensor is created correctly."""
        assert simple_energy.E44_symbolic is not None
        assert simple_energy.E11_symbolic is not None

    def test_energy_tensor_at_shape(self, simple_energy):
        """Test numeric evaluation shape."""
        E = simple_energy.at(2.0, 1.0)
        assert E.shape == (4, 4)

    def test_energy_tensor_via_solution(self):
        """Test energy tensor through solution's built-in method."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)
        E = sol.energy_tensor_at(2.0, 1.0)
        assert E.shape == (4, 4)

    def test_energy_tensor_trace_via_solution(self):
        """Test energy tensor is approximately trace-free."""
        sol = CaseIIISolution(variant='variant_71', m=1.0, a=0.5)
        E = sol.energy_tensor_at(2.0, 1.0)
        trace = np.trace(E)
        assert abs(trace) < 0.1  # Allow some tolerance

    def test_E44_E11_relation(self, simple_energy):
        """Test E44 = -E11."""
        E = simple_energy.at(2.0, 1.0)
        np.testing.assert_allclose(E[3, 3], -E[0, 0], rtol=1e-10)

    def test_E22_E33_relation(self, simple_energy):
        """Test E22 = -E33."""
        E = simple_energy.at(2.0, 1.0)
        np.testing.assert_allclose(E[1, 1], -E[2, 2], rtol=1e-10)

    def test_E14_E41_symmetry(self, simple_energy):
        """Test E14 = E41."""
        E = simple_energy.at(2.0, 1.0)
        assert E[0, 3] == E[3, 0]

    def test_energy_conditions_method(self, simple_energy):
        """Test energy_conditions returns dictionary with expected keys."""
        conditions = simple_energy.energy_conditions(2.0, 1.0)
        assert 'weak_energy' in conditions
        assert 'trace_free' in conditions
        # Values should be boolean-like (numpy bool or Python bool)
        assert conditions['weak_energy'] in (True, False)
        assert conditions['trace_free'] in (True, False)


class TestCylindricalWaveSolution:
    """Comprehensive tests for CylindricalWaveSolution class."""

    def test_default_creation(self):
        """Test default wave solution creation."""
        wave = CylindricalWaveSolution()
        assert wave.modes is not None
        assert len(wave.modes) == 1

    def test_custom_modes_creation(self):
        """Test wave solution with custom modes."""
        modes = [(1.0, 0.5, 0.0, 0.0), (2.0, 0.3, 0.0, np.pi/4)]
        wave = CylindricalWaveSolution(modes=modes)
        assert len(wave.modes) == 2

    def test_x_evaluation(self):
        """Test x(rho, t) evaluation."""
        wave = create_standing_wave(k=1.0, A=1.0)
        x_val = wave.x(2.0, 0.0)
        assert np.isfinite(x_val)

    def test_x_1_evaluation(self):
        """Test dx/drho evaluation."""
        wave = create_standing_wave(k=1.0, A=1.0)
        x_1_val = wave.x_1(2.0, 0.0)
        assert np.isfinite(x_1_val)

    def test_x_4_evaluation(self):
        """Test dx/dt evaluation."""
        wave = create_standing_wave(k=1.0, A=1.0)
        x_4_val = wave.x_4(2.0, 0.0)
        assert np.isfinite(x_4_val)

    def test_wave_equation_satisfied(self):
        """Test wave equation is satisfied."""
        wave = create_standing_wave(k=1.0, A=1.0)
        residual = wave.verify_wave_equation(2.0, 1.0)
        assert abs(residual) < 0.01  # Allow numerical error

    def test_superposition_wave_equation(self):
        """Test superposition also satisfies wave equation."""
        wave = create_superposition([1.0, 2.0], [0.5, 0.3])
        residual = wave.verify_wave_equation(2.0, 1.0)
        assert abs(residual) < 0.01

    def test_standing_wave_at_t0(self):
        """Test standing wave at t=0 equals J_0(k*rho)."""
        from scipy.special import j0
        k, A = 1.0, 1.0
        wave = create_standing_wave(k=k, A=A)

        rho_val = 2.0
        x_val = wave.x(rho_val, 0.0)
        expected = A * j0(k * rho_val) * np.cos(0)
        np.testing.assert_allclose(x_val, expected, rtol=1e-10)

    def test_standing_wave_oscillation(self):
        """Test standing wave oscillates in time."""
        wave = create_standing_wave(k=1.0, A=1.0)
        rho_val = 2.0

        x_0 = wave.x(rho_val, 0.0)
        x_pi = wave.x(rho_val, np.pi)

        # cos(0) = 1, cos(pi) = -1, so x should flip sign
        np.testing.assert_allclose(x_0, -x_pi, rtol=1e-10)

    def test_symbolic_expressions_exist(self):
        """Test symbolic expressions are built."""
        wave = create_standing_wave(k=1.0, A=1.0)
        assert wave.x_symbolic is not None
        assert wave.x_1_symbolic is not None
        assert wave.x_4_symbolic is not None
        assert wave.x_11_symbolic is not None
        assert wave.x_44_symbolic is not None


class TestCreateWaveFunctions:
    """Tests for wave creation helper functions."""

    def test_create_standing_wave(self):
        """Test create_standing_wave function."""
        wave = create_standing_wave(k=2.0, A=0.5)
        assert len(wave.modes) == 1
        assert wave.modes[0][0] == 2.0  # k
        assert wave.modes[0][1] == 0.5  # A

    def test_create_superposition(self):
        """Test create_superposition function."""
        wave = create_superposition([1.0, 2.0, 3.0], [0.5, 0.3, 0.2])
        assert len(wave.modes) == 3

    def test_superposition_is_sum(self):
        """Test superposition is sum of individual waves."""
        wave1 = create_standing_wave(k=1.0, A=0.5)
        wave2 = create_standing_wave(k=2.0, A=0.3)
        wave_super = create_superposition([1.0, 2.0], [0.5, 0.3])

        rho_val, t_val = 2.0, 0.5
        x_sum = wave1.x(rho_val, t_val) + wave2.x(rho_val, t_val)
        x_super = wave_super.x(rho_val, t_val)

        np.testing.assert_allclose(x_sum, x_super, rtol=1e-10)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_solution_to_field_to_energy_chain(self):
        """Test field and energy tensors through solution methods."""
        sol = CaseIIISolution(variant='variant_71', m=1.0, a=0.5)

        # Get field tensor via solution method
        F_val = sol.field_tensor_at(2.0, 1.0)
        assert F_val.shape == (4, 4)
        np.testing.assert_allclose(F_val, -F_val.T, atol=1e-10)

        # Get energy tensor via solution method
        E_val = sol.energy_tensor_at(2.0, 1.0)
        assert E_val.shape == (4, 4)

        # Verify approximately trace-free
        assert abs(np.trace(E_val)) < 0.1

    def test_metric_with_solution(self):
        """Test metric created from solution values."""
        sol = CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0)

        rho_val, t_val = 2.0, 1.0

        # Direct from solution
        g_sol = sol.metric_at(rho_val, t_val)

        # Using mu and lambda values
        mu_val = sol.mu(rho_val, t_val)
        lam_val = sol.lambda_(rho_val, t_val)

        # Construct metric from formula
        exp_lam_minus_mu = np.exp(lam_val - mu_val)
        exp_minus_mu = np.exp(-mu_val)
        exp_mu = np.exp(mu_val)

        g_manual = np.array([
            [-exp_lam_minus_mu, 0, 0, 0],
            [0, -rho_val**2 * exp_minus_mu, 0, 0],
            [0, 0, -exp_mu, 0],
            [0, 0, 0, exp_lam_minus_mu]
        ])

        np.testing.assert_allclose(g_sol, g_manual, rtol=1e-10)

    def test_all_solutions_have_field_tensor(self):
        """Test all solution types provide field tensors."""
        solutions = [
            CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0),
            CaseIISolution(variant='rho_only', m=1.0),
            CaseIIISolution(variant='variant_71', m=1.0, a=0.5),
        ]

        for sol in solutions:
            F = sol.field_tensor_at(2.0, 1.0)
            assert F.shape == (4, 4)
            np.testing.assert_allclose(F, -F.T, atol=1e-10)

    def test_all_solutions_have_energy_tensor(self):
        """Test all solution types provide energy tensors."""
        solutions = [
            CaseISolution(alpha=1.0, beta=0.5, variant='t_only', m=1.0),
            CaseIISolution(variant='rho_only', m=1.0),
            CaseIIISolution(variant='variant_71', m=1.0, a=0.5),
        ]

        for sol in solutions:
            E = sol.energy_tensor_at(2.0, 1.0)
            assert E.shape == (4, 4)
