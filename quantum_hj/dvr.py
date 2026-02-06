"""
Discrete Variable Representation (DVR) solver for 2D quantum systems.

This module provides reference eigenvalues and wavefunctions using the
sinc-DVR method, which is numerically exact for smooth potentials and
serves as validation for PINN-based solvers.

Reference:
    Colbert, D. T. & Miller, W. H. (1992). "A novel discrete variable
    representation for quantum mechanical reactive scattering via the
    S-matrix Kohn method." J. Chem. Phys. 96, 1982-1991.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


class DVRSolver2D:
    """
    Discrete Variable Representation solver for 2D potentials.

    Uses sinc-DVR basis functions for numerically exact reference
    eigenvalues. The method discretizes position space on a grid and
    represents the kinetic energy operator using sinc-function matrix
    elements.

    Parameters
    ----------
    potential : callable or Potential2D
        Potential function V(x, y)
    x_range : tuple
        (x_min, x_max) range for x coordinate
    y_range : tuple
        (y_min, y_max) range for y coordinate
    n_basis_x : int
        Number of DVR basis functions in x (default: 80)
    n_basis_y : int
        Number of DVR basis functions in y (default: 80)
    mass : float
        Particle mass (default: 1.0)
    hbar : float
        Reduced Planck constant (default: 1.0)

    Attributes
    ----------
    x_grid : ndarray
        DVR grid points in x
    y_grid : ndarray
        DVR grid points in y
    dx : float
        Grid spacing in x
    dy : float
        Grid spacing in y
    energies : ndarray or None
        Computed eigenvalues (after calling solve())
    wavefunctions : ndarray or None
        Computed eigenvectors (after calling solve())

    Example
    -------
    >>> from quantum_hj.potentials_2d import HarmonicOscillator2D
    >>> pot = HarmonicOscillator2D(omega_x=1.0, omega_y=1.0)
    >>> dvr = DVRSolver2D(pot, x_range=(-6, 6), y_range=(-6, 6), n_basis_x=60)
    >>> energies, wavefunctions = dvr.solve(n_states=10)
    >>> print(energies[:3])  # Should be close to [1.0, 2.0, 2.0]
    """

    def __init__(self, potential, x_range, y_range, n_basis_x=80, n_basis_y=80,
                 mass=1.0, hbar=1.0):
        self.potential = potential
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.n_basis_x = n_basis_x
        self.n_basis_y = n_basis_y
        self.mass = mass
        self.hbar = hbar

        # Compute grid spacing
        self.dx = (self.x_max - self.x_min) / (n_basis_x + 1)
        self.dy = (self.y_max - self.y_min) / (n_basis_y + 1)

        # Create DVR grid points (interior points only)
        self.x_grid = np.linspace(self.x_min + self.dx, self.x_max - self.dx, n_basis_x)
        self.y_grid = np.linspace(self.y_min + self.dy, self.y_max - self.dy, n_basis_y)

        # Storage for results
        self.energies = None
        self.wavefunctions = None
        self._hamiltonian = None

    def _kinetic_matrix_1d(self, n, dx):
        """
        Compute 1D sinc-DVR kinetic energy matrix.

        The matrix elements are:
            T_{ij} = (hbar^2 / 2m) * (pi^2 / 3dx^2)           if i == j
            T_{ij} = (hbar^2 / 2m) * (-1)^{i-j} * 2/(dx^2 * (i-j)^2)  if i != j

        Parameters
        ----------
        n : int
            Number of grid points
        dx : float
            Grid spacing

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse kinetic energy matrix
        """
        prefactor = self.hbar**2 / (2 * self.mass * dx**2)

        # Diagonal elements: pi^2 / 3
        diag = np.full(n, np.pi**2 / 3)

        # Build sparse matrix with diagonals
        diagonals = [diag]
        offsets = [0]

        # Off-diagonal elements: (-1)^k * 2 / k^2 for k = i - j
        for k in range(1, n):
            off_diag_val = ((-1)**k) * 2.0 / k**2
            off_diag = np.full(n - k, off_diag_val)
            diagonals.append(off_diag)
            diagonals.append(off_diag)  # Symmetric
            offsets.append(k)
            offsets.append(-k)

        T = sparse.diags(diagonals, offsets, shape=(n, n), format='csr')

        return prefactor * T

    def build_hamiltonian(self):
        """
        Build the full 2D Hamiltonian matrix.

        H = T_x ⊗ I_y + I_x ⊗ T_y + V

        where ⊗ denotes the Kronecker product, T_x and T_y are kinetic
        energy matrices, I_x and I_y are identity matrices, and V is
        the diagonal potential matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse Hamiltonian matrix of size (n_x * n_y, n_x * n_y)
        """
        n_x = self.n_basis_x
        n_y = self.n_basis_y

        # 1D kinetic energy matrices
        T_x = self._kinetic_matrix_1d(n_x, self.dx)
        T_y = self._kinetic_matrix_1d(n_y, self.dy)

        # Identity matrices
        I_x = sparse.eye(n_x, format='csr')
        I_y = sparse.eye(n_y, format='csr')

        # Kinetic energy: T_x ⊗ I_y + I_x ⊗ T_y
        T_total = sparse.kron(T_x, I_y, format='csr') + sparse.kron(I_x, T_y, format='csr')

        # Potential energy matrix (diagonal)
        # Grid ordering: (x_0, y_0), (x_0, y_1), ..., (x_0, y_{n_y-1}), (x_1, y_0), ...
        V_diag = np.zeros(n_x * n_y)
        for i, x in enumerate(self.x_grid):
            for j, y in enumerate(self.y_grid):
                idx = i * n_y + j
                V_diag[idx] = self.potential(x, y)

        V = sparse.diags(V_diag, 0, format='csr')

        # Full Hamiltonian
        H = T_total + V

        self._hamiltonian = H
        return H

    def solve(self, n_states=20, sigma=None):
        """
        Compute eigenvalues and eigenvectors.

        Uses ARPACK sparse eigenvalue solver (scipy.sparse.linalg.eigsh)
        to find the lowest n_states eigenvalues.

        Parameters
        ----------
        n_states : int
            Number of lowest eigenvalues to compute (default: 20)
        sigma : float or None
            Target energy for shift-invert mode. If None, finds lowest
            eigenvalues. If specified, finds eigenvalues near sigma.

        Returns
        -------
        energies : ndarray
            Array of n_states eigenvalues, sorted ascending
        wavefunctions : ndarray
            Array of shape (n_x * n_y, n_states) containing eigenvectors
        """
        if self._hamiltonian is None:
            self.build_hamiltonian()

        # Solve eigenvalue problem
        if sigma is not None:
            # Shift-invert mode for finding eigenvalues near sigma
            energies, wavefunctions = eigsh(
                self._hamiltonian,
                k=n_states,
                sigma=sigma,
                which='LM'
            )
        else:
            # Find smallest eigenvalues
            energies, wavefunctions = eigsh(
                self._hamiltonian,
                k=n_states,
                which='SM'
            )

        # Sort by energy (eigsh may not return sorted results)
        idx = np.argsort(energies)
        energies = energies[idx]
        wavefunctions = wavefunctions[:, idx]

        self.energies = energies
        self.wavefunctions = wavefunctions

        return energies, wavefunctions

    def get_wavefunction_2d(self, state_index):
        """
        Reshape a 1D eigenvector to 2D grid.

        Parameters
        ----------
        state_index : int
            Index of the state (0 = ground state)

        Returns
        -------
        ndarray
            Wavefunction on 2D grid of shape (n_basis_x, n_basis_y)
        """
        if self.wavefunctions is None:
            raise ValueError("Must call solve() first")

        psi_1d = self.wavefunctions[:, state_index]
        psi_2d = psi_1d.reshape(self.n_basis_x, self.n_basis_y)

        return psi_2d

    def probability_density(self, state_index):
        """
        Compute probability density |ψ|² for a given state.

        Parameters
        ----------
        state_index : int
            Index of the state (0 = ground state)

        Returns
        -------
        ndarray
            Probability density on 2D grid
        """
        psi_2d = self.get_wavefunction_2d(state_index)
        return np.abs(psi_2d)**2

    def expectation_value(self, operator, state_index):
        """
        Compute expectation value <ψ|O|ψ> for a diagonal operator.

        Parameters
        ----------
        operator : callable
            Function O(x, y) representing the operator
        state_index : int
            Index of the state

        Returns
        -------
        float
            Expectation value
        """
        psi_2d = self.get_wavefunction_2d(state_index)

        # Build operator matrix on grid
        op_vals = np.zeros((self.n_basis_x, self.n_basis_y))
        for i, x in enumerate(self.x_grid):
            for j, y in enumerate(self.y_grid):
                op_vals[i, j] = operator(x, y)

        # Expectation value: sum over grid (with dx*dy normalization implicit in DVR)
        return np.sum(np.abs(psi_2d)**2 * op_vals) * self.dx * self.dy


def compute_dvr_reference(potential, x_range, y_range, n_states=20,
                          n_basis=60, mass=1.0, hbar=1.0):
    """
    Convenience function to compute DVR reference energies.

    Parameters
    ----------
    potential : callable or Potential2D
        Potential function V(x, y)
    x_range : tuple
        (x_min, x_max) range
    y_range : tuple
        (y_min, y_max) range
    n_states : int
        Number of states to compute
    n_basis : int
        Number of basis functions per dimension
    mass : float
        Particle mass
    hbar : float
        Reduced Planck constant

    Returns
    -------
    energies : ndarray
        Array of eigenvalues
    """
    dvr = DVRSolver2D(
        potential, x_range, y_range,
        n_basis_x=n_basis, n_basis_y=n_basis,
        mass=mass, hbar=hbar
    )
    energies, _ = dvr.solve(n_states=n_states)
    return energies
