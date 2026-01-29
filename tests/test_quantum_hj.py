"""
Tests for the Quantum Hamilton-Jacobi solver.
"""

import pytest
import numpy as np
from quantum_hj import QuantumHJSolver, HarmonicOscillator, Potential


class TestHarmonicOscillator:
    """Tests for the HarmonicOscillator potential."""

    def test_potential_value(self):
        """V(x) = x²/2 in natural units."""
        ho = HarmonicOscillator()
        assert ho(0) == 0
        assert ho(1) == 0.5
        assert ho(2) == 2.0
        assert ho(-1) == 0.5

    def test_potential_with_parameters(self):
        """V(x) = (1/2) m ω² x²"""
        ho = HarmonicOscillator(omega=2.0, mass=0.5)
        # V(x) = 0.5 * 0.5 * 4 * x² = x²
        assert ho(1) == 1.0
        assert ho(2) == 4.0

    def test_turning_points(self):
        """Turning points at x = ±√(2E)"""
        ho = HarmonicOscillator()
        tp = ho.turning_points(E=2.0)
        assert tp is not None
        assert np.isclose(tp[0], -2.0)
        assert np.isclose(tp[1], 2.0)

    def test_turning_points_zero_energy(self):
        """No turning points for E ≤ 0."""
        ho = HarmonicOscillator()
        assert ho.turning_points(E=0) is None
        assert ho.turning_points(E=-1) is None

    def test_exact_energies(self):
        """E_n = n + 0.5 in natural units."""
        ho = HarmonicOscillator()
        exact = ho.exact_energies(n_max=5)
        expected = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
        np.testing.assert_allclose(exact, expected)


class TestQuantumHJSolver:
    """Tests for the QuantumHJSolver."""

    @pytest.fixture
    def solver(self):
        """Create solver with harmonic oscillator potential."""
        return QuantumHJSolver(HarmonicOscillator())

    def test_ground_state(self, solver):
        """E_0 should be close to 0.5."""
        E0 = solver.find_energy(n=0)
        assert np.isclose(E0, 0.5, rtol=0.01)

    def test_first_excited(self, solver):
        """E_1 should be close to 1.5."""
        E1 = solver.find_energy(n=1)
        assert np.isclose(E1, 1.5, rtol=0.01)

    def test_multiple_energies(self, solver):
        """Test first 5 energy levels."""
        energies = solver.find_energies(n_max=4)
        expected = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        np.testing.assert_allclose(energies, expected, rtol=0.01)

    def test_ten_levels(self, solver):
        """Test first 10 energy levels within 1% accuracy."""
        energies = solver.find_energies(n_max=9)
        expected = np.array([n + 0.5 for n in range(10)])
        np.testing.assert_allclose(energies, expected, rtol=0.01)

    def test_validation(self, solver):
        """Validation should pass for accurate energies."""
        energies = solver.find_energies(n_max=5)
        result = solver.validate(energies)
        assert result['status'] == 'pass'
        assert result['max_relative_error'] < 0.01

    def test_quantization_condition_at_eigenvalue(self, solver):
        """J/ℏ should equal n + 0.5 at eigenvalues."""
        # At E = 0.5 (ground state), J/ℏ should be 0.5
        J_over_hbar = solver._quantization_condition(E=0.5)
        assert np.isclose(J_over_hbar, 0.5, rtol=0.01)

    def test_energy_ordering(self, solver):
        """Energies should be monotonically increasing."""
        energies = solver.find_energies(n_max=9)
        assert all(energies[i] < energies[i+1] for i in range(len(energies)-1))

    def test_energy_spacing(self, solver):
        """Energy spacing should be approximately ℏω = 1."""
        energies = solver.find_energies(n_max=9)
        spacings = np.diff(energies)
        np.testing.assert_allclose(spacings, np.ones(9), rtol=0.02)


class TestContourIntegration:
    """Tests for contour integration internals."""

    @pytest.fixture
    def solver(self):
        return QuantumHJSolver(HarmonicOscillator())

    def test_contour_shape(self, solver):
        """Contour should have correct number of points."""
        contour = solver._create_contour(E=1.0)
        assert len(contour) == solver.n_points

    def test_contour_is_closed_ellipse(self, solver):
        """Contour should be a closed ellipse around turning points."""
        contour = solver._create_contour(E=1.0)
        # Contour should have points in both upper and lower half-planes
        has_upper = any(c.imag > 0 for c in contour)
        has_lower = any(c.imag < 0 for c in contour)
        assert has_upper and has_lower

    def test_contour_encloses_turning_points(self, solver):
        """Contour should enclose the turning points."""
        E = 2.0
        contour = solver._create_contour(E)
        tp = solver.potential.turning_points(E)

        # Real part of contour should extend beyond turning points
        real_min = min(c.real for c in contour)
        real_max = max(c.real for c in contour)
        assert real_min < tp[0]
        assert real_max > tp[1]


class TestCustomPotential:
    """Test that the base class can be extended."""

    def test_custom_potential_interface(self):
        """Custom potentials should work with the solver."""

        class QuarticOscillator(Potential):
            def __call__(self, x):
                return x**4 / 4

            def turning_points(self, E):
                if E <= 0:
                    return None
                x_tp = (4 * E) ** 0.25
                return (-x_tp, x_tp)

        solver = QuantumHJSolver(QuarticOscillator())
        # Should be able to compute energy (won't validate exact value)
        E0 = solver.find_energy(n=0, E_min=0.1, E_max=2.0)
        assert E0 > 0
