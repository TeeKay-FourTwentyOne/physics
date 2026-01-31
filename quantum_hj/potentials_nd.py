"""
N-dimensional potential classes for the quantum Hamilton-Jacobi solver.

This module provides N-dimensional potentials with normal mode analysis
for validating and training the N-D PINN solver.
"""

import numpy as np
from abc import ABC, abstractmethod


class PotentialND(ABC):
    """Abstract base class for N-dimensional quantum potentials."""

    @property
    @abstractmethod
    def ndim(self):
        """Return the number of dimensions."""
        pass

    @abstractmethod
    def __call__(self, *coords):
        """
        Evaluate V(x_1, x_2, ..., x_n).

        Parameters
        ----------
        coords : float or array-like
            Coordinates (can be complex for PINN evaluation)

        Returns
        -------
        float or array
            Potential value(s)
        """
        pass

    @abstractmethod
    def turning_points(self, E, dim, fixed_coords):
        """
        Find classical turning points in dimension `dim` at fixed other coordinates.

        Parameters
        ----------
        E : float
            Total energy
        dim : int
            Dimension index (0-based)
        fixed_coords : dict
            Dictionary mapping dimension index to fixed coordinate value.
            Dimensions not in dict use 0.0.

        Returns
        -------
        tuple or None
            (left, right) turning points, or None if none exist
        """
        pass

    def exact_energy(self, quantum_numbers, hbar=1.0):
        """
        Return exact energy if known analytically.

        Parameters
        ----------
        quantum_numbers : tuple
            (n_0, n_1, ..., n_{d-1}) quantum numbers
        hbar : float
            Reduced Planck constant

        Returns
        -------
        float or None
            Exact energy or None if unknown
        """
        return None

    def hessian_at_minimum(self):
        """
        Return the Hessian matrix at the potential minimum.

        For quadratic potentials, this gives the coefficient matrix.
        For anharmonic potentials, this gives the local curvature.

        Returns
        -------
        ndarray
            d x d Hessian matrix
        """
        return None

    def normal_mode_frequencies(self, mass=1.0):
        """
        Compute normal mode frequencies by diagonalizing the Hessian.

        Returns
        -------
        ndarray
            Array of normal mode frequencies omega_i
        """
        H = self.hessian_at_minimum()
        if H is None:
            return None

        # Hessian eigenvalues are omega^2 for unit mass
        eigenvalues = np.linalg.eigvalsh(H)

        # Ensure positive (bound potential)
        if np.any(eigenvalues <= 0):
            raise ValueError("Potential has non-positive curvature (unbounded)")

        return np.sqrt(eigenvalues / mass)

    def normal_mode_transformation(self, mass=1.0):
        """
        Compute the transformation matrix to normal mode coordinates.

        Returns
        -------
        tuple
            (omega, R) where omega are frequencies and R is the rotation matrix
            such that R diagonalizes the Hessian: R^T H R = diag(omega^2)
        """
        H = self.hessian_at_minimum()
        if H is None:
            return None, None

        eigenvalues, R = np.linalg.eigh(H)

        if np.any(eigenvalues <= 0):
            raise ValueError("Potential has non-positive curvature")

        omega = np.sqrt(eigenvalues / mass)
        return omega, R


class IsotropicOscillatorND(PotentialND):
    """
    N-dimensional isotropic harmonic oscillator.

    V = (1/2) * sum_i(x_i^2)

    Exact energy: E = hbar * omega * (sum(n_i) + ndim/2)

    Parameters
    ----------
    ndim : int
        Number of dimensions
    omega : float
        Angular frequency (same for all dimensions)
    mass : float
        Particle mass
    """

    def __init__(self, ndim, omega=1.0, mass=1.0):
        self._ndim = ndim
        self.omega = omega
        self.mass = mass

    @property
    def ndim(self):
        return self._ndim

    def __call__(self, *coords):
        """V = (1/2) * m * omega^2 * sum(x_i^2)"""
        if len(coords) != self._ndim:
            raise ValueError(f"Expected {self._ndim} coordinates, got {len(coords)}")

        result = 0.0
        for x in coords:
            result = result + x**2

        return 0.5 * self.mass * self.omega**2 * result

    def turning_points(self, E, dim, fixed_coords=None):
        """
        Classical turning points in dimension `dim`.

        For isotropic oscillator: x_dim = +/- sqrt(2*E_avail / (m*omega^2))
        where E_avail = E - V(other coords)
        """
        if fixed_coords is None:
            fixed_coords = {}

        # Energy used by other dimensions
        E_other = 0.0
        for d in range(self._ndim):
            if d != dim:
                x_d = fixed_coords.get(d, 0.0)
                E_other += 0.5 * self.mass * self.omega**2 * x_d**2

        E_avail = E - E_other
        if E_avail <= 0:
            return None

        x_tp = np.sqrt(2 * E_avail / (self.mass * self.omega**2))
        return (-x_tp, x_tp)

    def exact_energy(self, quantum_numbers, hbar=1.0):
        """
        E = hbar * omega * (sum(n_i) + ndim/2)
        """
        if len(quantum_numbers) != self._ndim:
            raise ValueError(f"Expected {self._ndim} quantum numbers")

        return hbar * self.omega * (sum(quantum_numbers) + self._ndim / 2.0)

    def hessian_at_minimum(self):
        """For isotropic oscillator, Hessian is diagonal with m*omega^2."""
        return self.mass * self.omega**2 * np.eye(self._ndim)

    def alpha_matrix(self):
        """
        Return the coefficient matrix for exact quantum momentum.

        For isotropic oscillator: p_i = -i * alpha * x_i
        where alpha = sqrt(m * omega) = omega (for m=1)

        Returns
        -------
        ndarray
            Diagonal matrix with alpha values
        """
        alpha = np.sqrt(self.mass) * self.omega
        return alpha * np.eye(self._ndim)


class CoupledOscillator3D(PotentialND):
    """
    3D coupled harmonic oscillator with asymmetric coupling.

    V = (1/2)(x^2 + y^2 + z^2) + lambda1*xy + lambda2*yz

    The Hessian matrix is:
        H = [[1, lambda1, 0],
             [lambda1, 1, lambda2],
             [0, lambda2, 1]]

    This is a tridiagonal matrix whose eigenvalues give normal mode frequencies.

    Parameters
    ----------
    lambda1 : float
        x-y coupling strength (default: 0.2)
    lambda2 : float
        y-z coupling strength (default: 0.15)
    mass : float
        Particle mass (default: 1.0)

    Notes
    -----
    For bounded potential, the Hessian must be positive definite.
    This requires the eigenvalues to be positive.
    """

    def __init__(self, lambda1=0.2, lambda2=0.15, mass=1.0):
        self._lambda1 = lambda1
        self._lambda2 = lambda2
        self.mass = mass

        # Compute and cache normal mode analysis
        H = self.hessian_at_minimum()
        eigenvalues = np.linalg.eigvalsh(H)
        if np.any(eigenvalues <= 0):
            raise ValueError(
                f"Coupling too strong: Hessian has eigenvalues {eigenvalues}. "
                "Reduce lambda1 and/or lambda2."
            )

        self._omega = np.sqrt(eigenvalues / mass)
        _, self._R = np.linalg.eigh(H)

    @property
    def ndim(self):
        return 3

    @property
    def lambda1(self):
        return self._lambda1

    @property
    def lambda2(self):
        return self._lambda2

    def __call__(self, *coords):
        """V = (1/2)(x^2 + y^2 + z^2) + lambda1*xy + lambda2*yz"""
        if len(coords) != 3:
            raise ValueError(f"Expected 3 coordinates, got {len(coords)}")

        x, y, z = coords
        return (0.5 * (x**2 + y**2 + z**2) +
                self._lambda1 * x * y +
                self._lambda2 * y * z)

    def turning_points(self, E, dim, fixed_coords=None):
        """
        Classical turning points in dimension `dim`.

        Solve V = E for x_dim given other coordinates.
        """
        if fixed_coords is None:
            fixed_coords = {}

        # Get fixed coordinates (default to 0)
        coords = [fixed_coords.get(d, 0.0) for d in range(3)]

        if dim == 0:  # Solve for x
            y, z = coords[1], coords[2]
            # V = (1/2)x^2 + lambda1*x*y + const = E
            # x^2 + 2*lambda1*y*x + (y^2 + z^2 + 2*lambda2*y*z - 2E) = 0
            a = 0.5
            b = self._lambda1 * y
            c = 0.5 * (y**2 + z**2) + self._lambda2 * y * z - E

            disc = b**2 - 4 * a * c
            if disc < 0:
                return None
            sqrt_disc = np.sqrt(disc)
            x_left = (-b - sqrt_disc) / (2 * a)
            x_right = (-b + sqrt_disc) / (2 * a)
            return (x_left, x_right)

        elif dim == 1:  # Solve for y
            x, z = coords[0], coords[2]
            # V = (1/2)y^2 + (lambda1*x + lambda2*z)*y + const = E
            a = 0.5
            b = self._lambda1 * x + self._lambda2 * z
            c = 0.5 * (x**2 + z**2) - E

            disc = b**2 - 4 * a * c
            if disc < 0:
                return None
            sqrt_disc = np.sqrt(disc)
            y_left = (-b - sqrt_disc) / (2 * a)
            y_right = (-b + sqrt_disc) / (2 * a)
            return (y_left, y_right)

        else:  # dim == 2, solve for z
            x, y = coords[0], coords[1]
            # V = (1/2)z^2 + lambda2*y*z + const = E
            a = 0.5
            b = self._lambda2 * y
            c = 0.5 * (x**2 + y**2) + self._lambda1 * x * y - E

            disc = b**2 - 4 * a * c
            if disc < 0:
                return None
            sqrt_disc = np.sqrt(disc)
            z_left = (-b - sqrt_disc) / (2 * a)
            z_right = (-b + sqrt_disc) / (2 * a)
            return (z_left, z_right)

    def hessian_at_minimum(self):
        """
        Hessian matrix at minimum (origin).

        H = [[1, lambda1, 0],
             [lambda1, 1, lambda2],
             [0, lambda2, 1]]
        """
        return np.array([
            [1.0, self._lambda1, 0.0],
            [self._lambda1, 1.0, self._lambda2],
            [0.0, self._lambda2, 1.0]
        ])

    def exact_energy(self, quantum_numbers, hbar=1.0):
        """
        Exact energy using normal mode frequencies.

        E = hbar * sum_i(omega_i * (n_i + 1/2))

        where omega_i are the normal mode frequencies (sorted ascending)
        and n_i are quantum numbers in normal mode basis.
        """
        if len(quantum_numbers) != 3:
            raise ValueError("Expected 3 quantum numbers")

        omega_sorted = np.sort(self._omega)
        return hbar * sum(omega_sorted[i] * (quantum_numbers[i] + 0.5)
                         for i in range(3))

    def alpha_matrix(self):
        """
        Return the coefficient matrix for exact quantum momentum.

        For coupled oscillator, the momentum in Cartesian coordinates is:
            p = -i * alpha_matrix @ coords

        where alpha_matrix = R @ diag(omega) @ R^T
        """
        omega_diag = np.diag(self._omega)
        return self._R @ omega_diag @ self._R.T


class CoupledOscillator4D(PotentialND):
    """
    4D coupled harmonic oscillator with block-diagonal coupling.

    V = (1/2)(x^2 + y^2 + z^2 + w^2) + lambda*(xy + zw)

    The Hessian matrix is block-diagonal:
        H = [[1, lambda, 0, 0],
             [lambda, 1, 0, 0],
             [0, 0, 1, lambda],
             [0, 0, lambda, 1]]

    This has eigenvalues: 1+lambda, 1-lambda, 1+lambda, 1-lambda
    So normal mode frequencies are:
        omega_plus = sqrt(1 + lambda)
        omega_minus = sqrt(1 - lambda)
    each with multiplicity 2.

    Parameters
    ----------
    coupling : float
        Coupling strength lambda (default: 0.2). Must satisfy |lambda| < 1.
    mass : float
        Particle mass (default: 1.0)
    """

    def __init__(self, coupling=0.2, mass=1.0):
        if abs(coupling) >= 1.0:
            raise ValueError("Coupling must satisfy |lambda| < 1 for bounded potential")

        self._coupling = coupling
        self.mass = mass

        # Normal mode frequencies
        self.omega_plus = np.sqrt(1.0 + coupling)
        self.omega_minus = np.sqrt(1.0 - coupling)

    @property
    def ndim(self):
        return 4

    @property
    def coupling(self):
        return self._coupling

    def __call__(self, *coords):
        """V = (1/2)(x^2 + y^2 + z^2 + w^2) + lambda*(xy + zw)"""
        if len(coords) != 4:
            raise ValueError(f"Expected 4 coordinates, got {len(coords)}")

        x, y, z, w = coords
        return (0.5 * (x**2 + y**2 + z**2 + w**2) +
                self._coupling * (x * y + z * w))

    def turning_points(self, E, dim, fixed_coords=None):
        """
        Classical turning points in dimension `dim`.
        """
        if fixed_coords is None:
            fixed_coords = {}

        coords = [fixed_coords.get(d, 0.0) for d in range(4)]

        if dim == 0:  # Solve for x
            y, z, w = coords[1], coords[2], coords[3]
            # V = (1/2)x^2 + lambda*x*y + const = E
            a = 0.5
            b = self._coupling * y
            c = 0.5 * (y**2 + z**2 + w**2) + self._coupling * z * w - E

            disc = b**2 - 4 * a * c
            if disc < 0:
                return None
            sqrt_disc = np.sqrt(disc)
            return ((-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a))

        elif dim == 1:  # Solve for y
            x, z, w = coords[0], coords[2], coords[3]
            a = 0.5
            b = self._coupling * x
            c = 0.5 * (x**2 + z**2 + w**2) + self._coupling * z * w - E

            disc = b**2 - 4 * a * c
            if disc < 0:
                return None
            sqrt_disc = np.sqrt(disc)
            return ((-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a))

        elif dim == 2:  # Solve for z
            x, y, w = coords[0], coords[1], coords[3]
            a = 0.5
            b = self._coupling * w
            c = 0.5 * (x**2 + y**2 + w**2) + self._coupling * x * y - E

            disc = b**2 - 4 * a * c
            if disc < 0:
                return None
            sqrt_disc = np.sqrt(disc)
            return ((-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a))

        else:  # dim == 3, solve for w
            x, y, z = coords[0], coords[1], coords[2]
            a = 0.5
            b = self._coupling * z
            c = 0.5 * (x**2 + y**2 + z**2) + self._coupling * x * y - E

            disc = b**2 - 4 * a * c
            if disc < 0:
                return None
            sqrt_disc = np.sqrt(disc)
            return ((-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a))

    def hessian_at_minimum(self):
        """
        Block-diagonal Hessian matrix.

        H = [[1, lambda, 0, 0],
             [lambda, 1, 0, 0],
             [0, 0, 1, lambda],
             [0, 0, lambda, 1]]
        """
        lam = self._coupling
        return np.array([
            [1.0, lam, 0.0, 0.0],
            [lam, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, lam],
            [0.0, 0.0, lam, 1.0]
        ])

    def exact_energy(self, quantum_numbers, hbar=1.0):
        """
        Exact energy using normal mode frequencies.

        Due to block structure, we have two pairs of modes with frequencies
        omega_plus and omega_minus.

        For quantum numbers (n_0, n_1, n_2, n_3) in Cartesian basis,
        the normal mode quantum numbers are:
        - (n_0 + n_1) excitations in the (x+y, x-y) subspace
        - (n_2 + n_3) excitations in the (z+w, z-w) subspace

        For the ground state, E = hbar * (omega_plus + omega_minus)

        More generally, we need the normal mode basis.
        """
        if len(quantum_numbers) != 4:
            raise ValueError("Expected 4 quantum numbers")

        # Sorted frequencies: omega_minus < omega_plus
        # Degeneracy: each appears twice
        # Sort quantum numbers to match (assuming sorted assignment)
        n_sorted = sorted(quantum_numbers)

        # Assign to modes: lowest n values to lowest frequency
        # omega_minus appears twice, omega_plus appears twice
        E = 0.0
        E += self.omega_minus * (n_sorted[0] + 0.5) * hbar
        E += self.omega_minus * (n_sorted[1] + 0.5) * hbar
        E += self.omega_plus * (n_sorted[2] + 0.5) * hbar
        E += self.omega_plus * (n_sorted[3] + 0.5) * hbar

        return E

    def alpha_matrix(self):
        """
        Return the coefficient matrix for exact quantum momentum.

        For block-diagonal coupling:
            p_x = -i * (a*x + b*y)
            p_y = -i * (b*x + a*y)
            p_z = -i * (a*z + b*w)
            p_w = -i * (b*z + a*w)

        where a = (omega_plus + omega_minus) / 2
              b = (omega_plus - omega_minus) / 2
        """
        a = (self.omega_plus + self.omega_minus) / 2
        b = (self.omega_plus - self.omega_minus) / 2

        return np.array([
            [a, b, 0, 0],
            [b, a, 0, 0],
            [0, 0, a, b],
            [0, 0, b, a]
        ])
