"""
2D potential classes for the quantum Hamilton-Jacobi solver.

This module provides 2D separable potentials with exact solutions
for validating the 2D PINN solver.
"""

import numpy as np
from abc import ABC, abstractmethod


class Potential2D(ABC):
    """Abstract base class for 2D quantum potentials."""

    @abstractmethod
    def __call__(self, x, y):
        """
        Evaluate V(x, y).

        Parameters
        ----------
        x : float or array-like
            x-coordinate (can be complex)
        y : float or array-like
            y-coordinate (can be complex)

        Returns
        -------
        float or array
            Potential value(s)
        """
        pass

    @abstractmethod
    def turning_points_x(self, E, y=0.0):
        """
        Find classical turning points in x at fixed y where V(x, y) = E.

        Parameters
        ----------
        E : float
            Total energy
        y : float
            Fixed y-coordinate (default: 0)

        Returns
        -------
        tuple or None
            (x_left, x_right) turning points, or None if none exist
        """
        pass

    @abstractmethod
    def turning_points_y(self, E, x=0.0):
        """
        Find classical turning points in y at fixed x where V(x, y) = E.

        Parameters
        ----------
        E : float
            Total energy
        x : float
            Fixed x-coordinate (default: 0)

        Returns
        -------
        tuple or None
            (y_left, y_right) turning points, or None if none exist
        """
        pass

    def exact_energies(self, n_x_max, n_y_max):
        """
        Return exact energies for validation (if known).

        Parameters
        ----------
        n_x_max : int
            Maximum quantum number in x
        n_y_max : int
            Maximum quantum number in y

        Returns
        -------
        dict or None
            Dictionary mapping (n_x, n_y) to energy, or None if unknown
        """
        return None


class HarmonicOscillator2D(Potential2D):
    """
    2D Harmonic oscillator potential: V(x, y) = (1/2) m (ω_x² x² + ω_y² y²)

    This is a separable potential, allowing independent action integrals
    in x and y directions.

    Parameters
    ----------
    omega_x : float
        Angular frequency in x direction (default: 1.0)
    omega_y : float
        Angular frequency in y direction (default: 1.0)
    mass : float
        Particle mass (default: 1.0)

    Notes
    -----
    The exact energies are:
        E_{n_x, n_y} = ℏω_x(n_x + 1/2) + ℏω_y(n_y + 1/2)

    For the ground state with ω_x = ω_y = 1:
        E_{0,0} = 1.0

    The exact quantum momenta are:
        p_x(x) = -i√(mω_x) x = -ix  (for m=ω_x=1)
        p_y(y) = -i√(mω_y) y = -iy  (for m=ω_y=1)
    """

    def __init__(self, omega_x=1.0, omega_y=1.0, mass=1.0):
        self.omega_x = omega_x
        self.omega_y = omega_y
        self.mass = mass

    def __call__(self, x, y):
        """V(x, y) = (1/2) m (ω_x² x² + ω_y² y²)"""
        return (0.5 * self.mass * self.omega_x**2 * x**2 +
                0.5 * self.mass * self.omega_y**2 * y**2)

    def V_x(self, x):
        """Potential contribution from x only."""
        return 0.5 * self.mass * self.omega_x**2 * x**2

    def V_y(self, y):
        """Potential contribution from y only."""
        return 0.5 * self.mass * self.omega_y**2 * y**2

    def turning_points_x(self, E, y=0.0):
        """
        Classical turning points in x at fixed y.

        For separable potential: E_x = E - V_y(y), then
        V_x(x) = E_x gives turning points.

        Parameters
        ----------
        E : float
            Total energy
        y : float
            Fixed y-coordinate

        Returns
        -------
        tuple or None
            (x_left, x_right) or None if E_x <= 0
        """
        # Energy available for x motion
        E_x = E - self.V_y(y)
        if E_x <= 0:
            return None
        # Turning points: V_x(x) = E_x => x = ±√(2E_x / mω_x²)
        x_tp = np.sqrt(2 * E_x / (self.mass * self.omega_x**2))
        return (-x_tp, x_tp)

    def turning_points_y(self, E, x=0.0):
        """
        Classical turning points in y at fixed x.

        Parameters
        ----------
        E : float
            Total energy
        x : float
            Fixed x-coordinate

        Returns
        -------
        tuple or None
            (y_left, y_right) or None if E_y <= 0
        """
        # Energy available for y motion
        E_y = E - self.V_x(x)
        if E_y <= 0:
            return None
        # Turning points: V_y(y) = E_y => y = ±√(2E_y / mω_y²)
        y_tp = np.sqrt(2 * E_y / (self.mass * self.omega_y**2))
        return (-y_tp, y_tp)

    def exact_energies(self, n_x_max, n_y_max, hbar=1.0):
        """
        Exact bound state energies for the 2D harmonic oscillator.

        E_{n_x, n_y} = ℏω_x(n_x + 1/2) + ℏω_y(n_y + 1/2)

        Parameters
        ----------
        n_x_max : int
            Maximum quantum number in x
        n_y_max : int
            Maximum quantum number in y
        hbar : float
            Reduced Planck constant (default: 1.0)

        Returns
        -------
        dict
            Dictionary mapping (n_x, n_y) tuple to energy value
        """
        energies = {}
        for n_x in range(n_x_max + 1):
            for n_y in range(n_y_max + 1):
                E = (hbar * self.omega_x * (n_x + 0.5) +
                     hbar * self.omega_y * (n_y + 0.5))
                energies[(n_x, n_y)] = E
        return energies

    def exact_energy(self, n_x, n_y, hbar=1.0):
        """
        Exact energy for a single state.

        Parameters
        ----------
        n_x : int
            Quantum number in x
        n_y : int
            Quantum number in y
        hbar : float
            Reduced Planck constant

        Returns
        -------
        float
            Energy value
        """
        return (hbar * self.omega_x * (n_x + 0.5) +
                hbar * self.omega_y * (n_y + 0.5))

    def exact_momentum_x(self, x):
        """
        Exact quantum momentum in x for ground state.

        For ω_x = m = 1: p_x = -ix

        Parameters
        ----------
        x : complex or array
            Position(s) in x

        Returns
        -------
        complex or array
            Momentum p_x(x)
        """
        # p_x = -i * sqrt(m * omega_x) * x
        return -1j * np.sqrt(self.mass * self.omega_x) * x

    def exact_momentum_y(self, y):
        """
        Exact quantum momentum in y for ground state.

        For ω_y = m = 1: p_y = -iy

        Parameters
        ----------
        y : complex or array
            Position(s) in y

        Returns
        -------
        complex or array
            Momentum p_y(y)
        """
        # p_y = -i * sqrt(m * omega_y) * y
        return -1j * np.sqrt(self.mass * self.omega_y) * y

    def exact_action_x(self, n_x, hbar=1.0):
        """
        Exact action integral in x direction.

        J_x = ℏ(n_x + 1/2)

        Parameters
        ----------
        n_x : int
            Quantum number in x
        hbar : float
            Reduced Planck constant

        Returns
        -------
        float
            Action J_x
        """
        return hbar * (n_x + 0.5)

    def exact_action_y(self, n_y, hbar=1.0):
        """
        Exact action integral in y direction.

        J_y = ℏ(n_y + 1/2)

        Parameters
        ----------
        n_y : int
            Quantum number in y
        hbar : float
            Reduced Planck constant

        Returns
        -------
        float
            Action J_y
        """
        return hbar * (n_y + 0.5)


class CoupledHarmonicOscillator(Potential2D):
    """
    Coupled 2D Harmonic oscillator: V(x, y) = (1/2)(x² + y²) + λxy

    This is a non-separable potential where the coupling term λxy
    mixes the x and y degrees of freedom. Despite being non-separable,
    this potential can be diagonalized by rotating to normal mode
    coordinates, giving exact eigenvalues:

        E_{n+, n-} = ℏω+(n+ + 1/2) + ℏω-(n- + 1/2)

    where ω± = √(1 ± λ) for the standard case with m = ω_x = ω_y = 1.

    Parameters
    ----------
    coupling : float
        Coupling strength λ (default: 0.2). Must satisfy |λ| < 1
        for the potential to be bounded below.
    mass : float
        Particle mass (default: 1.0)

    Raises
    ------
    ValueError
        If |coupling| >= 1 (potential unbounded below)
    """

    def __init__(self, coupling=0.2, mass=1.0):
        if abs(coupling) >= 1.0:
            raise ValueError("Coupling must satisfy |λ| < 1 for bounded potential")
        self.coupling = coupling
        self.mass = mass

        # Normal mode frequencies: ω± = √(1 ± λ)
        self.omega_plus = np.sqrt(1.0 + coupling)
        self.omega_minus = np.sqrt(1.0 - coupling)

    def __call__(self, x, y):
        """V(x, y) = (1/2)(x² + y²) + λxy"""
        return 0.5 * (x**2 + y**2) + self.coupling * x * y

    def turning_points_x(self, E, y=0.0):
        """
        Classical turning points in x at fixed y.

        Solves V(x, y) = E for x:
            (1/2)x² + λxy + (1/2)y² - E = 0
            x = -λy ± √(λ²y² - y² + 2E)

        Parameters
        ----------
        E : float
            Total energy
        y : float
            Fixed y-coordinate

        Returns
        -------
        tuple or None
            (x_left, x_right) or None if no real roots
        """
        # Discriminant: λ²y² - y² + 2E = (λ² - 1)y² + 2E
        disc = (self.coupling**2 - 1) * y**2 + 2 * E
        if disc < 0:
            return None
        sqrt_disc = np.sqrt(disc)
        x_left = -self.coupling * y - sqrt_disc
        x_right = -self.coupling * y + sqrt_disc
        return (x_left, x_right)

    def turning_points_y(self, E, x=0.0):
        """
        Classical turning points in y at fixed x.

        Solves V(x, y) = E for y:
            (1/2)y² + λxy + (1/2)x² - E = 0
            y = -λx ± √(λ²x² - x² + 2E)

        Parameters
        ----------
        E : float
            Total energy
        x : float
            Fixed x-coordinate

        Returns
        -------
        tuple or None
            (y_left, y_right) or None if no real roots
        """
        disc = (self.coupling**2 - 1) * x**2 + 2 * E
        if disc < 0:
            return None
        sqrt_disc = np.sqrt(disc)
        y_left = -self.coupling * x - sqrt_disc
        y_right = -self.coupling * x + sqrt_disc
        return (y_left, y_right)

    def exact_energies(self, n_plus_max, n_minus_max, hbar=1.0):
        """
        Exact eigenvalues in normal mode basis.

        E_{n+, n-} = ℏω+(n+ + 1/2) + ℏω-(n- + 1/2)

        Parameters
        ----------
        n_plus_max : int
            Maximum quantum number for ω+ mode
        n_minus_max : int
            Maximum quantum number for ω- mode
        hbar : float
            Reduced Planck constant

        Returns
        -------
        dict
            Dictionary mapping (n+, n-) to energy
        """
        energies = {}
        for n_plus in range(n_plus_max + 1):
            for n_minus in range(n_minus_max + 1):
                E = (hbar * self.omega_plus * (n_plus + 0.5) +
                     hbar * self.omega_minus * (n_minus + 0.5))
                energies[(n_plus, n_minus)] = E
        return energies

    def exact_energy(self, n_plus, n_minus, hbar=1.0):
        """
        Exact energy for a single normal mode state.

        Parameters
        ----------
        n_plus : int
            Quantum number for ω+ mode
        n_minus : int
            Quantum number for ω- mode
        hbar : float
            Reduced Planck constant

        Returns
        -------
        float
            Energy value
        """
        return (hbar * self.omega_plus * (n_plus + 0.5) +
                hbar * self.omega_minus * (n_minus + 0.5))

    def ground_state_energy(self, hbar=1.0):
        """
        Ground state energy.

        E_0 = (ℏ/2)(ω+ + ω-)

        Parameters
        ----------
        hbar : float
            Reduced Planck constant

        Returns
        -------
        float
            Ground state energy
        """
        return 0.5 * hbar * (self.omega_plus + self.omega_minus)


class HenonHeiles(Potential2D):
    """
    Henon-Heiles potential: V(x, y) = (1/2)(x² + y²) + λ(x²y - y³/3)

    A classic model for studying the transition from regular to chaotic
    motion in Hamiltonian systems. The potential has C3v symmetry with
    three saddle points forming an equilateral triangle.

    For E < E_escape = 1/(6λ²), motion is bounded. Above this energy,
    trajectories can escape to infinity.

    Parameters
    ----------
    coupling : float
        Coupling strength λ (default: 0.1)
    mass : float
        Particle mass (default: 1.0)

    Attributes
    ----------
    escape_energy : float
        Critical energy above which classical trajectories escape
    """

    def __init__(self, coupling=0.1, mass=1.0):
        self.coupling = coupling
        self.mass = mass

        # Escape energy: E_escape = 1/(6λ²)
        if coupling != 0:
            self.escape_energy = 1.0 / (6.0 * coupling**2)
        else:
            self.escape_energy = np.inf

    def __call__(self, x, y):
        """V(x, y) = (1/2)(x² + y²) + λ(x²y - y³/3)"""
        return (0.5 * (x**2 + y**2) +
                self.coupling * (x**2 * y - y**3 / 3.0))

    def turning_points_x(self, E, y=0.0):
        """
        Classical turning points in x at fixed y.

        At y = 0, reduces to harmonic: V(x, 0) = (1/2)x²
        Otherwise, solve (1/2)x² + λx²y = E - (1/2)y² + λy³/3

        Parameters
        ----------
        E : float
            Total energy
        y : float
            Fixed y-coordinate

        Returns
        -------
        tuple or None
            (x_left, x_right) or None if no real roots
        """
        # V(x, y) = (1/2)x² + λyx² + (1/2)y² - λy³/3
        # (1/2 + λy)x² = E - (1/2)y² + λy³/3
        coeff = 0.5 + self.coupling * y
        rhs = E - 0.5 * y**2 + self.coupling * y**3 / 3.0

        if coeff <= 0 or rhs < 0:
            return None

        x_tp = np.sqrt(rhs / coeff)
        return (-x_tp, x_tp)

    def turning_points_y(self, E, x=0.0):
        """
        Classical turning points in y at fixed x.

        At x = 0, V(0, y) = (1/2)y² - λy³/3
        This is a cubic, requiring numerical solution in general.

        For simplicity, return approximate turning points assuming
        small perturbation from harmonic.

        Parameters
        ----------
        E : float
            Total energy
        x : float
            Fixed x-coordinate

        Returns
        -------
        tuple or None
            (y_left, y_right) or None if no bounded region
        """
        # For small λ, use harmonic approximation
        E_y = E - 0.5 * x**2
        if E_y <= 0:
            return None

        # Harmonic approximation: y_tp ≈ √(2E_y)
        y_tp_approx = np.sqrt(2 * E_y)

        # Refine by checking actual potential values
        # For the Henon-Heiles, the potential rises faster for y > 0
        # and slower (eventually decreasing) for y < 0

        # Return conservative estimate
        return (-y_tp_approx, y_tp_approx * 0.8)

    def is_bounded(self, E):
        """
        Check if classical motion is bounded at energy E.

        Parameters
        ----------
        E : float
            Total energy

        Returns
        -------
        bool
            True if E < escape_energy
        """
        return E < self.escape_energy

    def saddle_points(self):
        """
        Return the three saddle points of the potential.

        Located at angles 0°, 120°, 240° from origin at
        distance r = 1/λ from origin.

        Returns
        -------
        list of tuple
            [(x1, y1), (x2, y2), (x3, y3)] saddle point coordinates
        """
        if self.coupling == 0:
            return []

        r = 1.0 / self.coupling
        angles = [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]
        return [(r * np.cos(theta), r * np.sin(theta)) for theta in angles]


class QuarticOscillator2D(Potential2D):
    """
    Anharmonic 2D oscillator: V(x, y) = (1/2)(x² + y²) + β(x⁴ + y⁴) + γx²y²

    A non-separable potential with quartic anharmonicity.

    Parameters
    ----------
    beta : float
        Quartic self-coupling (default: 0.1)
    gamma : float
        Cross quartic coupling (default: 0.1)
    mass : float
        Particle mass (default: 1.0)
    """

    def __init__(self, beta=0.1, gamma=0.1, mass=1.0):
        self.beta = beta
        self.gamma = gamma
        self.mass = mass

    def __call__(self, x, y):
        """V(x, y) = (1/2)(x² + y²) + β(x⁴ + y⁴) + γx²y²"""
        return (0.5 * (x**2 + y**2) +
                self.beta * (x**4 + y**4) +
                self.gamma * x**2 * y**2)

    def turning_points_x(self, E, y=0.0):
        """
        Classical turning points in x at fixed y.

        For general quartic, requires numerical root finding.
        Returns approximate harmonic turning points.
        """
        E_x = E - 0.5 * y**2 - self.beta * y**4
        if E_x <= 0:
            return None

        # Use harmonic approximation
        x_tp = np.sqrt(2 * E_x)
        return (-x_tp, x_tp)

    def turning_points_y(self, E, x=0.0):
        """
        Classical turning points in y at fixed x.
        """
        E_y = E - 0.5 * x**2 - self.beta * x**4
        if E_y <= 0:
            return None

        y_tp = np.sqrt(2 * E_y)
        return (-y_tp, y_tp)
