"""
Sparse N-dimensional Discrete Variable Representation (DVR) solver.

This module provides a memory-efficient DVR solver for arbitrary dimensions
using term-by-term Kronecker product construction to avoid massive
intermediate matrices.

Reference:
    Colbert, D. T. & Miller, W. H. (1992). "A novel discrete variable
    representation for quantum mechanical reactive scattering via the
    S-matrix Kohn method." J. Chem. Phys. 96, 1982-1991.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


class DVRSolverND:
    """
    Sparse N-dimensional DVR solver.

    Uses sinc-DVR basis functions with memory-efficient Kronecker
    construction that builds the kinetic energy term-by-term to
    avoid massive intermediate matrices.

    Parameters
    ----------
    potential : callable or PotentialND
        Potential function V(x_1, x_2, ..., x_n)
    ranges : list of tuple
        [(min, max), ...] for each dimension
    n_basis : int or list of int
        Number of DVR basis functions per dimension.
        If int, same for all dimensions.
    mass : float
        Particle mass (default: 1.0)
    hbar : float
        Reduced Planck constant (default: 1.0)

    Attributes
    ----------
    ndim : int
        Number of dimensions
    grids : list of ndarray
        DVR grid points for each dimension
    spacings : list of float
        Grid spacings for each dimension
    total_basis : int
        Total number of basis functions (product of n_basis)
    energies : ndarray or None
        Computed eigenvalues (after calling solve())
    wavefunctions : ndarray or None
        Computed eigenvectors (after calling solve())
    """

    def __init__(self, potential, ranges, n_basis, mass=1.0, hbar=1.0):
        self.potential = potential
        self.mass = mass
        self.hbar = hbar

        # Handle scalar n_basis
        self.ndim = len(ranges)
        if isinstance(n_basis, int):
            self.n_basis = [n_basis] * self.ndim
        else:
            self.n_basis = list(n_basis)
            if len(self.n_basis) != self.ndim:
                raise ValueError(
                    f"n_basis length {len(self.n_basis)} != ndim {self.ndim}"
                )

        # Store ranges
        self.ranges = ranges

        # Compute grid spacings and create grid points
        self.spacings = []
        self.grids = []
        for d in range(self.ndim):
            x_min, x_max = ranges[d]
            n = self.n_basis[d]
            dx = (x_max - x_min) / (n + 1)
            self.spacings.append(dx)
            # Interior grid points
            grid = np.linspace(x_min + dx, x_max - dx, n)
            self.grids.append(grid)

        # Total basis size
        self.total_basis = 1
        for n in self.n_basis:
            self.total_basis *= n

        # Storage for results
        self.energies = None
        self.wavefunctions = None
        self._hamiltonian = None

    def _kinetic_matrix_1d(self, n, dx):
        """
        Compute 1D sinc-DVR kinetic energy matrix.

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

    def build_hamiltonian(self, verbose=False):
        """
        Build the full N-D Hamiltonian matrix using memory-efficient construction.

        H = sum_i (I_0 ⊗ ... ⊗ T_i ⊗ ... ⊗ I_{n-1}) + V

        Each kinetic term is built without forming the full product at once.

        Parameters
        ----------
        verbose : bool
            Print progress information

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse Hamiltonian matrix
        """
        if verbose:
            print(f"Building {self.ndim}D Hamiltonian with {self.total_basis} basis functions")

        # Precompute 1D kinetic matrices and identity matrices
        T_1d = []
        I_1d = []
        for d in range(self.ndim):
            T_1d.append(self._kinetic_matrix_1d(self.n_basis[d], self.spacings[d]))
            I_1d.append(sparse.eye(self.n_basis[d], format='csr'))

        # Build kinetic energy term by term (memory efficient)
        H = sparse.csr_matrix((self.total_basis, self.total_basis), dtype=np.float64)

        for i in range(self.ndim):
            if verbose:
                print(f"  Adding kinetic term for dimension {i}")

            # Build: I_0 ⊗ I_1 ⊗ ... ⊗ T_i ⊗ ... ⊗ I_{n-1}
            term = sparse.csr_matrix([[1.0]])  # Start with scalar 1

            for j in range(self.ndim):
                if j == i:
                    term = sparse.kron(term, T_1d[j], format='csr')
                else:
                    term = sparse.kron(term, I_1d[j], format='csr')

            H = H + term
            del term  # Free memory

        if verbose:
            print("  Computing potential diagonal")

        # Potential energy matrix (diagonal)
        V_diag = self._compute_potential_diagonal()
        V = sparse.diags(V_diag, 0, format='csr')

        H = H + V

        self._hamiltonian = H

        if verbose:
            nnz = H.nnz
            mem_mb = (H.data.nbytes + H.indices.nbytes + H.indptr.nbytes) / (1024**2)
            print(f"  Hamiltonian: {H.shape[0]} x {H.shape[1]}, {nnz} nonzeros, {mem_mb:.1f} MB")

        return H

    def _compute_potential_diagonal(self):
        """
        Compute potential values at all grid points.

        Grid ordering: outer loop over first dimension, inner over last.
        Multi-index (i_0, i_1, ..., i_{n-1}) maps to linear index:
            idx = i_0 * (n_1 * n_2 * ... * n_{n-1}) + i_1 * (n_2 * ... * n_{n-1}) + ... + i_{n-1}

        Returns
        -------
        ndarray
            1D array of potential values
        """
        V_diag = np.zeros(self.total_basis)

        # Compute strides for multi-index to linear index conversion
        strides = [1]
        for d in range(self.ndim - 1, 0, -1):
            strides.insert(0, strides[0] * self.n_basis[d])

        # Iterate over all grid points
        indices = [0] * self.ndim

        for linear_idx in range(self.total_basis):
            # Get coordinates from indices
            coords = [self.grids[d][indices[d]] for d in range(self.ndim)]

            # Evaluate potential
            V_diag[linear_idx] = self.potential(*coords)

            # Increment multi-index (like odometer)
            for d in range(self.ndim - 1, -1, -1):
                indices[d] += 1
                if indices[d] < self.n_basis[d]:
                    break
                indices[d] = 0

        return V_diag

    def _linear_to_multi_index(self, linear_idx):
        """Convert linear index to multi-index."""
        indices = []
        remaining = linear_idx
        for d in range(self.ndim):
            # Compute stride for this dimension
            stride = 1
            for dd in range(d + 1, self.ndim):
                stride *= self.n_basis[dd]

            indices.append(remaining // stride)
            remaining = remaining % stride

        return tuple(indices)

    def _multi_to_linear_index(self, multi_idx):
        """Convert multi-index to linear index."""
        linear_idx = 0
        stride = 1
        for d in range(self.ndim - 1, -1, -1):
            linear_idx += multi_idx[d] * stride
            stride *= self.n_basis[d]
        return linear_idx

    def solve(self, n_states=20, sigma=None, verbose=False):
        """
        Compute eigenvalues and eigenvectors.

        Uses ARPACK sparse eigenvalue solver.

        Parameters
        ----------
        n_states : int
            Number of lowest eigenvalues to compute
        sigma : float or None
            Target energy for shift-invert mode
        verbose : bool
            Print progress information

        Returns
        -------
        energies : ndarray
            Array of eigenvalues, sorted ascending
        wavefunctions : ndarray
            Array of shape (total_basis, n_states) containing eigenvectors
        """
        if self._hamiltonian is None:
            self.build_hamiltonian(verbose=verbose)

        if verbose:
            print(f"Solving for {n_states} eigenvalues...")

        # Ensure we don't ask for more states than basis size allows
        max_states = self.total_basis - 2
        if n_states > max_states:
            n_states = max_states
            if verbose:
                print(f"  Reduced to {n_states} states (basis limit)")

        # Solve eigenvalue problem
        if sigma is not None:
            energies, wavefunctions = eigsh(
                self._hamiltonian,
                k=n_states,
                sigma=sigma,
                which='LM'
            )
        else:
            energies, wavefunctions = eigsh(
                self._hamiltonian,
                k=n_states,
                which='SM'
            )

        # Sort by energy
        idx = np.argsort(energies)
        energies = energies[idx]
        wavefunctions = wavefunctions[:, idx]

        self.energies = energies
        self.wavefunctions = wavefunctions

        if verbose:
            print(f"  Lowest energies: {energies[:min(5, len(energies))]}")

        return energies, wavefunctions

    def get_wavefunction_nd(self, state_index):
        """
        Reshape a 1D eigenvector to N-D grid.

        Parameters
        ----------
        state_index : int
            Index of the state (0 = ground state)

        Returns
        -------
        ndarray
            Wavefunction on N-D grid of shape (n_0, n_1, ..., n_{d-1})
        """
        if self.wavefunctions is None:
            raise ValueError("Must call solve() first")

        psi_1d = self.wavefunctions[:, state_index]
        shape = tuple(self.n_basis)
        psi_nd = psi_1d.reshape(shape)

        return psi_nd

    def memory_estimate_mb(self):
        """
        Estimate memory usage for the Hamiltonian matrix.

        Returns
        -------
        float
            Estimated memory in megabytes
        """
        # Estimate number of nonzeros
        # Kinetic: for each dimension, full matrix in that dim, diagonal in others
        nnz_kinetic = 0
        for d in range(self.ndim):
            # Full matrix in dimension d: n_d^2
            # Diagonal in others: product of (n_i for i != d)
            other_prod = self.total_basis // self.n_basis[d]
            nnz_kinetic += self.n_basis[d]**2 * other_prod

        # Potential: diagonal, total_basis nonzeros
        nnz_potential = self.total_basis

        nnz_total = nnz_kinetic + nnz_potential

        # CSR format: data (8 bytes), indices (4 bytes), indptr (4 bytes per row + 1)
        bytes_data = nnz_total * 8
        bytes_indices = nnz_total * 4
        bytes_indptr = (self.total_basis + 1) * 4

        return (bytes_data + bytes_indices + bytes_indptr) / (1024**2)


def compute_dvr_reference_nd(potential, ranges, n_states=20, n_basis=30,
                             mass=1.0, hbar=1.0, verbose=False):
    """
    Convenience function to compute DVR reference energies for N-D potentials.

    Parameters
    ----------
    potential : callable or PotentialND
        Potential function V(x_1, ..., x_n)
    ranges : list of tuple
        [(min, max), ...] for each dimension
    n_states : int
        Number of states to compute
    n_basis : int
        Number of basis functions per dimension
    mass : float
        Particle mass
    hbar : float
        Reduced Planck constant
    verbose : bool
        Print progress information

    Returns
    -------
    energies : ndarray
        Array of eigenvalues
    """
    dvr = DVRSolverND(
        potential, ranges, n_basis=n_basis,
        mass=mass, hbar=hbar
    )
    energies, _ = dvr.solve(n_states=n_states, verbose=verbose)
    return energies
