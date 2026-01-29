"""
Quantum Hamilton-Jacobi solver using the Leacock-Padgett formalism.

This module implements the quantum Hamilton-Jacobi method for finding
bound state energies. The approach uses contour integration of the
quantum momentum function p(x) in the complex plane.

Reference:
    Leacock, R. A. & Padgett, M. J. (1983). "Hamilton-Jacobi Theory
    and the Quantum Action Variable." Phys. Rev. Lett. 50, 3-6.
"""

import numpy as np
from abc import ABC, abstractmethod
from scipy.optimize import brentq


class Potential(ABC):
    """Abstract base class for quantum potentials."""

    @abstractmethod
    def __call__(self, x):
        """Evaluate V(x)."""
        pass

    @abstractmethod
    def turning_points(self, E):
        """Find classical turning points where V(x) = E."""
        pass

    def exact_energies(self, n_max):
        """Return exact energies for validation (if known)."""
        return None


class HarmonicOscillator(Potential):
    """
    1D Harmonic oscillator potential: V(x) = (1/2) m ω² x²

    In natural units (m = ω = 1): V(x) = x²/2
    """

    def __init__(self, omega=1.0, mass=1.0):
        self.omega = omega
        self.mass = mass

    def __call__(self, x):
        """V(x) = (1/2) m ω² x²"""
        return 0.5 * self.mass * self.omega**2 * x**2

    def turning_points(self, E):
        """Classical turning points: x = ±√(2E / mω²)"""
        if E <= 0:
            return None
        x_tp = np.sqrt(2 * E / (self.mass * self.omega**2))
        return (-x_tp, x_tp)

    def exact_energies(self, n_max):
        """E_n = ℏω(n + 1/2)"""
        return np.array([self.omega * (n + 0.5) for n in range(n_max + 1)])


class MorsePotential(Potential):
    """
    Morse potential: V(x) = D_e * (1 - e^(-a(x - x_e)))^2

    Parameters
    ----------
    D_e : float
        Well depth (dissociation energy)
    a : float
        Controls the width of the potential well
    x_e : float
        Equilibrium position (default: 0)
    mass : float
        Particle mass (default: 1.0, for natural units)

    The Morse potential supports a finite number of bound states:
        n_max = floor(sqrt(2*m*D_e)/(a*hbar) - 1/2)

    With hbar = m = 1:
        n_max = floor(sqrt(2*D_e)/a - 1/2)
    """

    def __init__(self, D_e, a, x_e=0.0, mass=1.0):
        self.D_e = D_e
        self.a = a
        self.x_e = x_e
        self.mass = mass
        # Characteristic frequency
        self.omega = a * np.sqrt(2 * D_e / mass)

    def __call__(self, x):
        """V(x) = D_e * (1 - e^(-a(x - x_e)))^2"""
        return self.D_e * (1 - np.exp(-self.a * (x - self.x_e)))**2

    def turning_points(self, E):
        """
        Classical turning points where V(x) = E.

        For 0 < E < D_e, there are two turning points.
        """
        if E <= 0 or E >= self.D_e:
            return None

        sqrt_ratio = np.sqrt(E / self.D_e)

        # Right turning point: x > x_e
        x_right = self.x_e - np.log(1 - sqrt_ratio) / self.a

        # Left turning point: x < x_e
        x_left = self.x_e - np.log(1 + sqrt_ratio) / self.a

        return (x_left, x_right)

    def max_bound_states(self, hbar=1.0):
        """
        Maximum number of bound states supported.

        n_max = floor(sqrt(2*m*D_e)/(a*hbar) - 1/2)
        """
        lambda_param = np.sqrt(2 * self.mass * self.D_e) / (self.a * hbar)
        return int(np.floor(lambda_param - 0.5))

    def exact_energies(self, n_max, hbar=1.0):
        """
        Exact bound state energies for the Morse potential.

        E_n = hbar*omega*(n + 1/2) - [hbar*omega*(n + 1/2)]^2 / (4*D_e)

        Only returns energies for valid bound states (E_n < D_e).
        """
        max_n = self.max_bound_states(hbar)
        actual_n_max = min(n_max, max_n)

        energies = []
        for n in range(actual_n_max + 1):
            term1 = hbar * self.omega * (n + 0.5)
            term2 = (hbar * self.omega * (n + 0.5))**2 / (4 * self.D_e)
            E_n = term1 - term2
            if E_n < self.D_e:
                energies.append(E_n)
            else:
                break

        return np.array(energies) if energies else None


class QuantumHJSolver:
    """
    Solver for quantum bound states using the Leacock-Padgett formalism.

    The quantum Hamilton-Jacobi equation leads to a quantization condition:

        J = (1/2πi) ∮ p(x) dx = ℏ(n + 1/2)

    where p(x) = √(2m(E - V(x))) is the classical momentum, and the
    contour encloses the classical turning points.

    For the harmonic oscillator, this integral can be evaluated using
    the residue theorem, giving the exact quantization condition.

    Parameters
    ----------
    potential : Potential
        The potential V(x)
    hbar : float
        Reduced Planck constant (default: 1.0)
    mass : float
        Particle mass (default: 1.0)
    """

    def __init__(self, potential, hbar=1.0, mass=1.0):
        self.potential = potential
        self.hbar = hbar
        self.mass = mass

        # Integration parameters
        self.n_points = 500  # Points along contour

    def _classical_momentum_squared(self, x, E):
        """
        Compute p²(x) = 2m(E - V(x)).

        This is the squared momentum; the actual momentum is ±√(p²).
        """
        V = self.potential(x)
        return 2 * self.mass * (E - V)

    def _create_contour(self, E):
        """
        Create a closed contour in the complex x-plane enclosing the turning points.

        Uses an ellipse that goes around both turning points, with the
        major axis along the real axis.
        """
        tp = self.potential.turning_points(E)
        if tp is None:
            raise ValueError(f"No turning points found for E = {E}")

        x_left, x_right = tp
        center = (x_left + x_right) / 2
        a = 1.2 * (x_right - x_left) / 2  # Semi-major axis (real direction)
        b = 0.5 * a  # Semi-minor axis (imaginary direction)

        # Parameterize: full ellipse, counterclockwise
        theta = np.linspace(0, 2 * np.pi, self.n_points, endpoint=False)
        x_contour = center + a * np.cos(theta) + 1j * b * np.sin(theta)

        return x_contour

    def _contour_integral(self, E):
        """
        Compute the contour integral J = (1/2πi) ∮ p dx.

        For the momentum p = √(2m(E-V)), we need to track the branch
        of the square root carefully as we traverse the contour.

        The key insight is that p² = 2m(E - V(x)), and for a harmonic
        oscillator V = mω²x²/2, we have:
            p² = 2m(E - mω²x²/2) = mω²(2E/mω² - x²) = mω²(a² - x²)
        where a = √(2E/mω²) is the turning point.

        So p = √(mω²) × √(a² - x²) = mω × √(a² - x²)

        The integral ∮ √(a² - x²) dx around a contour enclosing [-a, a]
        equals 2πi × Res = 2πi × (a²/2) = πi a²

        Actually, using the branch cut and residue theorem correctly:
        ∮ √(a² - x²) dx = ∮ √(a-x)√(a+x) dx

        The branch points are at x = ±a. Going around both gives
        ∮ p dx = -πi × a² × mω (for counterclockwise contour)
        """
        tp = self.potential.turning_points(E)
        if tp is None:
            raise ValueError(f"No turning points found for E = {E}")

        x_contour = self._create_contour(E)
        n = len(x_contour)

        # Compute p² at each point
        p_sq = np.array([self._classical_momentum_squared(x, E) for x in x_contour])

        # Take square root with proper branch tracking
        # Start with positive real part (classically allowed region)
        p_values = np.zeros(n, dtype=complex)

        # At the starting point (right side of ellipse, positive real axis region)
        # p should be real and positive for the classically allowed region
        p_values[0] = np.sqrt(p_sq[0] + 0j)

        # Since we start on the real axis to the right of the right turning point,
        # E - V > 0, so p² > 0 and p should be positive real
        # Choose branch: positive real momentum for x > x_right
        if p_values[0].real < 0:
            p_values[0] = -p_values[0]

        # Track the branch as we go around the contour
        for i in range(1, n):
            p_new = np.sqrt(p_sq[i] + 0j)
            # Choose sign to maintain continuity
            if np.abs(p_new - p_values[i-1]) > np.abs(-p_new - p_values[i-1]):
                p_new = -p_new
            p_values[i] = p_new

        # Compute ∮ p dx using trapezoidal rule for closed contour
        integral = 0j
        for i in range(n):
            dx = x_contour[(i + 1) % n] - x_contour[i]
            p_avg = (p_values[i] + p_values[(i + 1) % n]) / 2
            integral += p_avg * dx

        # J = (1/2πi) ∮ p dx
        # The result should be real for bound states
        J = integral / (2 * np.pi * 1j)

        return J

    def _quantization_condition(self, E):
        """
        Evaluate the quantization condition.

        Returns J/ℏ, which should equal (n + 1/2) at eigenvalues.

        Note: The sign depends on the contour orientation and branch choice.
        We use the absolute value since J should be positive for bound states.
        """
        J = self._contour_integral(E)
        # Take the imaginary part since our result comes out imaginary
        # due to branch cut structure
        return abs(J.imag) / self.hbar

    def find_energy(self, n, E_min=None, E_max=None, tol=1e-8):
        """
        Find the n-th eigenvalue using root finding.

        Parameters
        ----------
        n : int
            Quantum number (0 for ground state)
        E_min, E_max : float, optional
            Energy bracket for root finding
        tol : float
            Tolerance for energy convergence

        Returns
        -------
        float
            The n-th eigenvalue
        """
        target = n + 0.5  # Quantization: J/ℏ = n + 1/2

        # Default energy bracket based on expected spacing
        if E_min is None:
            E_min = max(0.01, n * 0.8)
        if E_max is None:
            E_max = (n + 1) * 1.5 + 0.5

        def f(E):
            return self._quantization_condition(E) - target

        # Find root using Brent's method
        E_n = brentq(f, E_min, E_max, xtol=tol)

        return E_n

    def find_energies(self, n_max):
        """
        Find eigenvalues E_0 through E_n_max.

        Parameters
        ----------
        n_max : int
            Maximum quantum number

        Returns
        -------
        numpy.ndarray
            Array of eigenvalues [E_0, E_1, ..., E_n_max]
        """
        energies = np.zeros(n_max + 1)

        for n in range(n_max + 1):
            # Use previous energy to guide bracket
            if n == 0:
                E_min, E_max = 0.01, 1.5
            else:
                E_min = energies[n - 1] + 0.3
                E_max = energies[n - 1] + 1.7

            energies[n] = self.find_energy(n, E_min, E_max)

        return energies

    def validate(self, energies):
        """
        Compare computed energies to exact values (if available).

        Parameters
        ----------
        energies : array-like
            Computed eigenvalues

        Returns
        -------
        dict
            Validation results with errors and status
        """
        exact = self.potential.exact_energies(len(energies) - 1)

        if exact is None:
            return {
                'status': 'unknown',
                'message': 'No exact solution available for comparison'
            }

        errors = np.abs(energies - exact)
        rel_errors = errors / exact

        return {
            'status': 'pass' if np.all(rel_errors < 0.01) else 'fail',
            'computed': energies,
            'exact': exact,
            'absolute_errors': errors,
            'relative_errors': rel_errors,
            'max_relative_error': np.max(rel_errors)
        }
