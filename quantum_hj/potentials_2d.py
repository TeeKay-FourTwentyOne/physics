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
