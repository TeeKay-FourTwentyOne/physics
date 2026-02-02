"""
N-dimensional quantum-number-conditioned PINN solver for quantum Hamilton-Jacobi equation.

This module implements an N-dimensional PINN approach where the network architecture
is explicitly conditioned on target quantum numbers (n_0, n_1, ..., n_{d-1}).
The key innovation is incorporating learnable pole positions that capture the
analytic structure of quantum momentum fields in arbitrary dimensions.

For state (n_0, n_1, ..., n_{d-1}), the momentum field has n_i poles in the
complex x_i-plane for each dimension i. These poles are initialized at Hermite
polynomial zeros and trained alongside the network weights and energy.

Reference:
    Leacock, R. A. & Padgett, M. J. (1983). "Hamilton-Jacobi/Action-Angle
    Quantum Mechanics." Phys. Rev. D 28, 2491-2502.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.special import roots_hermite

from .potentials_nd import (
    PotentialND, IsotropicOscillatorND,
    CoupledOscillator3D, CoupledOscillator4D,
    CoupledOscillator6D, TriatomicVibrational
)


def hermite_zeros(n):
    """
    Compute zeros of the n-th Hermite polynomial H_n(x).

    Parameters
    ----------
    n : int
        Order of Hermite polynomial (number of zeros)

    Returns
    -------
    ndarray
        Array of n zeros, sorted ascending
    """
    if n <= 0:
        return np.array([])
    zeros, _ = roots_hermite(n)
    return np.sort(zeros)


class QuantumNumberPINNND(nn.Module):
    """
    Neural network with learnable pole positions for state (n_0, n_1, ..., n_{d-1}).

    The momentum field ansatz is:
        p_i = NN_i(x_0, ..., x_{d-1}) + sum_{k=1}^{n_i} 1/(x_i - x_i^{(k)})

    where x_i^{(k)} are LEARNABLE parameters initialized at Hermite zeros.

    Parameters
    ----------
    quantum_numbers : tuple
        (n_0, n_1, ..., n_{d-1}) number of poles per dimension
    hidden_layers : int
        Number of hidden layers in smooth network (default: 4)
    hidden_size : int
        Neurons per hidden layer (default: 128)
    pole_scale : float
        Initial scale factor for pole positions (default: 1.0)
    """

    def __init__(self, quantum_numbers, hidden_layers=4, hidden_size=128, pole_scale=1.0):
        super().__init__()

        self.quantum_numbers = tuple(quantum_numbers)
        self.ndim = len(quantum_numbers)
        self.pole_scale = pole_scale

        # Learnable pole positions for each dimension
        self.poles = nn.ParameterList()
        for d in range(self.ndim):
            n_d = quantum_numbers[d]
            if n_d > 0:
                init_poles = hermite_zeros(n_d) * pole_scale
                self.poles.append(
                    nn.Parameter(torch.tensor(init_poles, dtype=torch.float32))
                )
            else:
                # No poles for ground state in this dimension
                # Register as buffer (not trainable)
                self.poles.append(nn.Parameter(torch.tensor([], dtype=torch.float32)))

        # Smooth part network: 2*ndim inputs (Re, Im for each coord)
        #                   -> 2*ndim outputs (Re, Im for each momentum)
        layers = []

        # Input layer
        layers.append(nn.Linear(2 * self.ndim, hidden_size))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(hidden_size, 2 * self.ndim))

        self.smooth_net = nn.Sequential(*layers)

        # Initialize output layer to small values
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        with torch.no_grad():
            last_layer = self.smooth_net[-1]
            last_layer.weight.mul_(0.1)
            last_layer.bias.fill_(0.0)

    def forward(self, *coords_complex):
        """
        Evaluate momentum (p_0, p_1, ..., p_{d-1}) for complex inputs.

        Parameters
        ----------
        coords_complex : torch.Tensor
            Complex tensors of coordinates, each of shape (batch_size,)

        Returns
        -------
        tuple of torch.Tensor
            (p_0, p_1, ..., p_{d-1}) as complex tensors
        """
        if len(coords_complex) != self.ndim:
            raise ValueError(f"Expected {self.ndim} coordinates, got {len(coords_complex)}")

        # Stack real and imaginary parts for network input
        inputs = []
        for x in coords_complex:
            inputs.append(x.real)
            inputs.append(x.imag)
        inputs = torch.stack(inputs, dim=-1)

        # Get smooth part from network
        output = self.smooth_net(inputs)

        # Extract smooth momentum components
        momenta = []
        for d in range(self.ndim):
            p_smooth = torch.complex(output[..., 2*d], output[..., 2*d + 1])

            # Add pole contributions
            n_d = self.quantum_numbers[d]
            if n_d > 0:
                x_d = coords_complex[d]
                for k in range(n_d):
                    pole_pos = self.poles[d][k].to(x_d.dtype)
                    p_smooth = p_smooth + 1.0 / (x_d - pole_pos)

            momenta.append(p_smooth)

        return tuple(momenta)

    def pole_positions(self):
        """
        Return current pole positions.

        Returns
        -------
        list of ndarray
            List of arrays containing pole positions for each dimension
        """
        positions = []
        for d in range(self.ndim):
            if self.quantum_numbers[d] > 0:
                positions.append(self.poles[d].detach().cpu().numpy())
            else:
                positions.append(np.array([]))
        return positions


class QuantumNumberTrainerND:
    """
    Train PINN for specific quantum numbers in N dimensions.

    This trainer jointly optimizes:
    - Network weights (smooth part of momentum)
    - Pole positions (shifted from Hermite zeros by coupling)
    - Energy E (driven to value satisfying quantization conditions)

    Parameters
    ----------
    potential : PotentialND
        N-dimensional potential function
    quantum_numbers : tuple
        Target quantum numbers (n_0, n_1, ..., n_{d-1})
    hbar : float
        Reduced Planck constant (default: 1.0)
    mass : float
        Particle mass (default: 1.0)
    hidden_layers : int or None
        Network hidden layers. If None, uses dimension-aware default (3 + ndim//2).
    hidden_size : int or None
        Neurons per hidden layer. If None, uses dimension-aware default (64*(ndim-1)).
    device : str
        PyTorch device ('cpu' or 'cuda')
    """

    def __init__(self, potential, quantum_numbers, hbar=1.0, mass=1.0,
                 hidden_layers=None, hidden_size=None, device=None):
        self.potential = potential
        self.quantum_numbers = tuple(quantum_numbers)
        self.ndim = len(quantum_numbers)
        self.hbar = hbar
        self.mass = mass

        # Dimension-aware defaults for network architecture
        # 3D: 4 layers, 128 neurons
        # 4D: 5 layers, 192 neurons
        # 5D: 6 layers, 320 neurons
        # 6D: 8 layers, 512 neurons
        if hidden_layers is None:
            if self.ndim >= 6:
                hidden_layers = 8
            else:
                hidden_layers = 3 + self.ndim // 2
        if hidden_size is None:
            if self.ndim >= 6:
                hidden_size = 512
            else:
                hidden_size = 64 * max(1, self.ndim - 1)

        if self.ndim != potential.ndim:
            raise ValueError(
                f"Quantum numbers dimension {self.ndim} != potential dimension {potential.ndim}"
            )

        # Auto-detect GPU
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Estimate initial energy
        E_init = float((sum(quantum_numbers) + self.ndim / 2) * hbar)

        # Estimate pole scale from energy
        pole_scale = np.sqrt(E_init) if E_init > 0 else 1.0

        # Create network
        self.net = QuantumNumberPINNND(
            quantum_numbers,
            hidden_layers=hidden_layers,
            hidden_size=hidden_size,
            pole_scale=pole_scale
        ).to(self.device)

        # Learnable energy parameter
        self.E = nn.Parameter(torch.tensor(E_init, dtype=torch.float32, device=self.device))

        # Training history
        self.loss_history = []
        self.physics_loss_history = []
        self.curl_loss_history = []
        self.quant_loss_history = []
        self.supervision_loss_history = []
        self.E_history = []

    def _potential_torch(self, *coords):
        """Evaluate potential at complex coordinates using PyTorch operations."""
        if isinstance(self.potential, IsotropicOscillatorND):
            result = 0.0
            for x in coords:
                result = result + x**2
            return 0.5 * self.potential.mass * self.potential.omega**2 * result

        elif isinstance(self.potential, CoupledOscillator3D):
            x, y, z = coords
            return (0.5 * (x**2 + y**2 + z**2) +
                    self.potential.lambda1 * x * y +
                    self.potential.lambda2 * y * z)

        elif isinstance(self.potential, CoupledOscillator4D):
            x, y, z, w = coords
            return (0.5 * (x**2 + y**2 + z**2 + w**2) +
                    self.potential.coupling * (x * y + z * w))

        elif isinstance(self.potential, CoupledOscillator6D):
            # V = (1/2)Σᵢxᵢ² + λΣᵢxᵢxᵢ₊₁ (periodic)
            result = 0.0
            for x in coords:
                result = result + 0.5 * x**2
            lam = self.potential.coupling
            for i in range(6):
                j = (i + 1) % 6
                result = result + lam * coords[i] * coords[j]
            return result

        elif isinstance(self.potential, TriatomicVibrational):
            r1, r2, theta1, theta2, phi1, phi2 = coords
            return (0.5 * self.potential.k_stretch * (r1**2 + r2**2) +
                    0.5 * self.potential.k_bend_in * (theta1**2 + theta2**2) +
                    0.5 * self.potential.k_bend_out * (phi1**2 + phi2**2) +
                    self.potential.lambda_sb * (r1 * theta1 + r2 * theta2))

        else:
            # Generic fallback for isotropic oscillator
            result = 0.0
            for x in coords:
                result = result + x**2
            return 0.5 * result

    def physics_loss(self, *coords_coll):
        """
        Compute physics loss enforcing the N-D quantum HJ equation.

        L = mean(|sum(p_i^2) + i*hbar*div(p) - 2m(E-V)|^2)

        where div(p) = sum_i dp_i/dx_i

        Parameters
        ----------
        coords_coll : torch.Tensor
            Complex collocation points for each dimension

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        h = 1e-4

        # Evaluate network at collocation points
        momenta = self.net(*coords_coll)

        # Compute divergence: sum of dp_i/dx_i
        divergence = torch.zeros_like(momenta[0])

        for d in range(self.ndim):
            # Create perturbed coordinates
            coords_plus = list(coords_coll)
            coords_minus = list(coords_coll)
            coords_plus[d] = coords_coll[d] + h
            coords_minus[d] = coords_coll[d] - h

            p_plus = self.net(*coords_plus)[d]
            p_minus = self.net(*coords_minus)[d]

            dp_d_dx_d = (p_plus - p_minus) / (2 * h)
            divergence = divergence + dp_d_dx_d

        # Compute p^2 = sum(p_i^2)
        p_squared = torch.zeros_like(momenta[0])
        for p in momenta:
            p_squared = p_squared + p**2

        # Evaluate potential
        V = self._potential_torch(*coords_coll)

        # Energy as complex
        E_complex = self.E.to(coords_coll[0].dtype)

        # QHJ residual
        residual = p_squared + 1j * self.hbar * divergence - 2 * self.mass * (E_complex - V)

        loss = torch.mean(torch.abs(residual)**2)

        return loss

    def curl_loss(self, *coords_coll):
        """
        Compute curl loss enforcing irrotationality.

        For p to be a gradient field: dp_i/dx_j = dp_j/dx_i for all i,j

        Number of constraints:
        - 2D: 1
        - 3D: 3 (xy, xz, yz)
        - 4D: 6 (xy, xz, xw, yz, yw, zw)
        - ND: ndim*(ndim-1)/2

        Parameters
        ----------
        coords_coll : torch.Tensor
            Complex collocation points

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        h = 1e-4
        loss = torch.tensor(0.0, device=self.device)

        # Loop over all pairs (i, j) with i < j
        for i in range(self.ndim):
            for j in range(i + 1, self.ndim):
                # Compute dp_i/dx_j
                coords_j_plus = list(coords_coll)
                coords_j_minus = list(coords_coll)
                coords_j_plus[j] = coords_coll[j] + h
                coords_j_minus[j] = coords_coll[j] - h

                p_i_j_plus = self.net(*coords_j_plus)[i]
                p_i_j_minus = self.net(*coords_j_minus)[i]
                dp_i_dx_j = (p_i_j_plus - p_i_j_minus) / (2 * h)

                # Compute dp_j/dx_i
                coords_i_plus = list(coords_coll)
                coords_i_minus = list(coords_coll)
                coords_i_plus[i] = coords_coll[i] + h
                coords_i_minus[i] = coords_coll[i] - h

                p_j_i_plus = self.net(*coords_i_plus)[j]
                p_j_i_minus = self.net(*coords_i_minus)[j]
                dp_j_dx_i = (p_j_i_plus - p_j_i_minus) / (2 * h)

                # Curl residual
                curl_ij = dp_i_dx_j - dp_j_dx_i
                loss = loss + torch.mean(torch.abs(curl_ij)**2)

        return loss

    def pole_regularization(self):
        """
        Regularization to keep poles in a reasonable region.

        Returns
        -------
        torch.Tensor
            Regularization loss
        """
        loss = torch.tensor(0.0, device=self.device)

        E_val = max(self.E.item(), 0.5)
        expected_scale = np.sqrt(2 * E_val)

        for d in range(self.ndim):
            if self.quantum_numbers[d] > 0:
                pole_sq = self.net.poles[d] ** 2
                loss = loss + torch.sum(torch.relu(pole_sq - expected_scale**2))

        return loss * 0.01

    def supervision_loss(self, n_points=100):
        """
        Supervision loss for momentum structure.

        For oscillator potentials, the exact quantum momentum is known.
        The network with poles needs to learn the smooth part correctly.

        Two types of supervision:
        1. Along each axis: supervise the active component (like 2D version)
        2. Bulk supervision at real points: enforce p_i = -i * (alpha @ coords)_i

        Parameters
        ----------
        n_points : int
            Number of supervision points per dimension

        Returns
        -------
        torch.Tensor
            Supervision loss
        """
        # Get alpha matrix if available
        if hasattr(self.potential, 'alpha_matrix'):
            alpha = self.potential.alpha_matrix()
        else:
            return torch.tensor(0.0, device=self.device)

        alpha_torch = torch.tensor(alpha, dtype=torch.float32, device=self.device)

        n_1d = int(n_points ** (1.0 / self.ndim))
        n_1d = max(n_1d, 5)

        loss = torch.tensor(0.0, device=self.device)

        # Part 1: Along each axis, supervise ONLY the active component (like 2D)
        for d in range(self.ndim):
            # Create grid along dimension d, others at zero
            if self.quantum_numbers[d] > 0:
                # Avoid poles
                x_left = torch.linspace(-3.0, -0.3, n_1d // 2, device=self.device)
                x_right = torch.linspace(0.3, 3.0, n_1d // 2, device=self.device)
                x_r = torch.cat([x_left, x_right])
            else:
                x_r = torch.linspace(-3.0, 3.0, n_1d, device=self.device)

            x_i = torch.linspace(-1.5, 1.5, n_1d, device=self.device)
            xr_grid, xi_grid = torch.meshgrid(x_r, x_i, indexing='ij')
            x_d = torch.complex(xr_grid.flatten(), xi_grid.flatten())

            # Build coordinate list with zeros in other dimensions
            coords = []
            for dd in range(self.ndim):
                if dd == d:
                    coords.append(x_d)
                else:
                    coords.append(torch.zeros_like(x_d))

            # Get network smooth output directly
            inputs = []
            for x in coords:
                inputs.append(x.real)
                inputs.append(x.imag)
            inputs = torch.stack(inputs, dim=-1)

            smooth_out = self.net.smooth_net(inputs)

            # Only supervise the active component p_d along axis d
            # p_d_smooth = -i * alpha[d,d] * x_d (diagonal element)
            p_smooth_actual = torch.complex(smooth_out[..., 2*d], smooth_out[..., 2*d + 1])
            p_smooth_expected = -1j * alpha_torch[d, d] * x_d

            loss = loss + torch.mean(torch.abs(p_smooth_actual - p_smooth_expected)**2)

        # Part 2: Coordinate plane supervision - enforce p_k = 0 when x_k = 0
        # This is CRITICAL for isotropic oscillators
        n_plane = n_1d
        coords_1d = torch.linspace(-2.0, 2.0, n_plane, device=self.device)

        for k in range(self.ndim):
            # Create points on the plane x_k = 0
            # All other coordinates vary, x_k = 0
            if self.ndim == 3:
                if k == 0:  # x = 0 plane
                    y_grid, z_grid = torch.meshgrid(coords_1d, coords_1d, indexing='ij')
                    plane_coords = [
                        torch.complex(torch.zeros_like(y_grid.flatten()), torch.zeros_like(y_grid.flatten())),
                        torch.complex(y_grid.flatten(), torch.zeros_like(y_grid.flatten())),
                        torch.complex(z_grid.flatten(), torch.zeros_like(z_grid.flatten()))
                    ]
                elif k == 1:  # y = 0 plane
                    x_grid, z_grid = torch.meshgrid(coords_1d, coords_1d, indexing='ij')
                    plane_coords = [
                        torch.complex(x_grid.flatten(), torch.zeros_like(x_grid.flatten())),
                        torch.complex(torch.zeros_like(x_grid.flatten()), torch.zeros_like(x_grid.flatten())),
                        torch.complex(z_grid.flatten(), torch.zeros_like(z_grid.flatten()))
                    ]
                else:  # z = 0 plane
                    x_grid, y_grid = torch.meshgrid(coords_1d, coords_1d, indexing='ij')
                    plane_coords = [
                        torch.complex(x_grid.flatten(), torch.zeros_like(x_grid.flatten())),
                        torch.complex(y_grid.flatten(), torch.zeros_like(y_grid.flatten())),
                        torch.complex(torch.zeros_like(x_grid.flatten()), torch.zeros_like(x_grid.flatten()))
                    ]
            elif self.ndim == 4:
                # For 4D, create 3D slices where x_k = 0
                # Increase density for 4D (was: max(3, n_plane // 2))
                n_4d = max(5, n_plane)
                coords_4d = torch.linspace(-2.0, 2.0, n_4d, device=self.device)
                other_dims = [i for i in range(4) if i != k]
                grids = torch.meshgrid(*[coords_4d] * 3, indexing='ij')
                plane_coords = []
                grid_idx = 0
                for i in range(4):
                    if i == k:
                        plane_coords.append(torch.complex(
                            torch.zeros_like(grids[0].flatten()),
                            torch.zeros_like(grids[0].flatten())
                        ))
                    else:
                        plane_coords.append(torch.complex(
                            grids[grid_idx].flatten(),
                            torch.zeros_like(grids[grid_idx].flatten())
                        ))
                        grid_idx += 1
            elif self.ndim >= 5:
                # For 5D+, use random sampling on hyperplane x_k = 0
                # Grid-based sampling is too memory-intensive
                n_samples = 500 if self.ndim == 5 else 1000  # More for 6D
                plane_coords = []
                for i in range(self.ndim):
                    if i == k:
                        # x_k = 0
                        plane_coords.append(torch.complex(
                            torch.zeros(n_samples, device=self.device),
                            torch.zeros(n_samples, device=self.device)
                        ))
                    else:
                        # Random coordinates in [-2, 2]
                        x_real = torch.rand(n_samples, device=self.device) * 4.0 - 2.0
                        plane_coords.append(torch.complex(
                            x_real,
                            torch.zeros(n_samples, device=self.device)
                        ))
            else:
                continue  # Skip for dimensions < 3

            # Get smooth network output at plane points
            plane_inputs = []
            for x in plane_coords:
                plane_inputs.append(x.real)
                plane_inputs.append(x.imag)
            plane_inputs = torch.stack(plane_inputs, dim=-1)

            plane_smooth_out = self.net.smooth_net(plane_inputs)

            # At x_k = 0 plane, for isotropic oscillator:
            # p_k should be -i * alpha[k,k] * 0 = 0
            # Other p_j should be -i * alpha[j,j] * x_j (diagonal alpha)
            for dd in range(self.ndim):
                p_smooth_actual = torch.complex(
                    plane_smooth_out[..., 2*dd], plane_smooth_out[..., 2*dd + 1]
                )

                # Expected: p_dd = -i * sum_j(alpha[dd, j] * coords_j)
                p_smooth_expected = torch.zeros_like(plane_coords[0])
                for j in range(self.ndim):
                    p_smooth_expected = p_smooth_expected - 1j * alpha_torch[dd, j] * plane_coords[j]

                loss = loss + torch.mean(torch.abs(p_smooth_actual - p_smooth_expected)**2)

        return loss

    def quantization_loss(self, n_points=50):
        """
        Loss driving energy to satisfy quantization conditions.

        J_i = (n_i + 0.5)*hbar for each dimension

        Parameters
        ----------
        n_points : int
            Number of points for action integral

        Returns
        -------
        torch.Tensor
            Quantization loss
        """
        loss = torch.tensor(0.0, device=self.device)

        for d in range(self.ndim):
            target = (self.quantum_numbers[d] + 0.5) * self.hbar
            J_d = self._compute_action(d, n_points, allow_grad=True)
            loss = loss + (J_d - target)**2

        return loss

    def _compute_action(self, dim, n_points=50, allow_grad=False):
        """
        Compute action integral J_dim using contour integration.

        Parameters
        ----------
        dim : int
            Dimension index
        n_points : int
            Number of contour points
        allow_grad : bool
            If True, allow gradients to flow

        Returns
        -------
        torch.Tensor
            Action integral J_dim (real scalar)
        """
        E_val = max(self.E.item(), 0.1)

        # Estimate turning points
        tp_estimate = np.sqrt(2 * E_val)

        # Create elliptical contour
        center = 0.0
        a = 1.3 * tp_estimate  # Semi-major axis (real)
        b = 0.4 * tp_estimate  # Semi-minor axis (imaginary)

        theta = torch.linspace(0, 2 * np.pi, n_points, device=self.device, dtype=torch.float32)
        x_real = center + a * torch.cos(theta)
        x_imag = b * torch.sin(theta)

        x_contour = torch.complex(x_real, x_imag)

        # Build coordinate list: contour in dimension dim, zeros elsewhere
        coords = []
        for dd in range(self.ndim):
            if dd == dim:
                coords.append(x_contour)
            else:
                coords.append(torch.zeros_like(x_contour))

        # Evaluate momentum on contour
        if allow_grad:
            momenta = self.net(*coords)
        else:
            with torch.no_grad():
                momenta = self.net(*coords)

        p_d = momenta[dim]

        # Trapezoidal integration: oint p_d dx_d
        integral = torch.tensor(0j, dtype=torch.complex64, device=self.device)
        for i in range(n_points):
            dx = x_contour[(i + 1) % n_points] - x_contour[i]
            p_avg = (p_d[i] + p_d[(i + 1) % n_points]) / 2
            integral = integral + p_avg * dx

        # J = (1/2*pi*i) * oint p dx
        J = integral / (2 * np.pi * 1j)

        return J.real

    def sample_collocation_points(self, n_points, imag_scale=0.3, importance_sampling=None):
        """
        Sample collocation points in the N-D complex space.

        For 6D and higher, uses importance sampling to concentrate points
        near regions of physical significance:
        - Origin (ground state density peak)
        - Classical turning surface (E = V boundary)
        - Coordinate axes (symmetry)

        Parameters
        ----------
        n_points : int
            Number of points to sample
        imag_scale : float
            Scale for imaginary parts
        importance_sampling : bool or None
            If None, auto-enable for ndim >= 6

        Returns
        -------
        tuple of torch.Tensor
            Complex collocation points for each dimension
        """
        E_val = max(self.E.item(), 0.5)
        tp = np.sqrt(2 * E_val)
        r = tp * 2.5

        # Auto-enable importance sampling for 6D+
        if importance_sampling is None:
            importance_sampling = (self.ndim >= 6)

        if not importance_sampling:
            # Standard uniform sampling
            coords = []
            for d in range(self.ndim):
                x_real = torch.rand(n_points, device=self.device) * 2 * r - r
                x_imag = torch.rand(n_points, device=self.device) * 2 * tp * imag_scale - tp * imag_scale
                coords.append(torch.complex(x_real, x_imag))
            return tuple(coords)

        # Importance sampling for 6D+
        # Split points into three regions:
        # 40% near origin (Gaussian), 40% near turning surface, 20% along axes
        n_origin = int(0.4 * n_points)
        n_turning = int(0.4 * n_points)
        n_axes = n_points - n_origin - n_turning

        all_coords = [[] for _ in range(self.ndim)]

        # Region 1: Near origin (Gaussian distribution)
        for d in range(self.ndim):
            x_real = torch.randn(n_origin, device=self.device) * (tp * 0.5)
            x_imag = torch.randn(n_origin, device=self.device) * (tp * imag_scale * 0.5)
            all_coords[d].append(torch.complex(x_real, x_imag))

        # Region 2: Near classical turning surface (shell at r ~ tp)
        # Sample radius near tp, direction uniform on sphere
        radii = torch.abs(torch.randn(n_turning, device=self.device) * 0.3 + 1.0) * tp
        # Generate uniform direction by normalizing Gaussian vectors
        directions = torch.randn(n_turning, self.ndim, device=self.device)
        norms = torch.sqrt(torch.sum(directions**2, dim=1, keepdim=True))
        directions = directions / (norms + 1e-8)

        for d in range(self.ndim):
            x_real = radii * directions[:, d]
            x_imag = torch.randn(n_turning, device=self.device) * (tp * imag_scale * 0.3)
            all_coords[d].append(torch.complex(x_real, x_imag))

        # Region 3: Along coordinate axes (sparse in all but one dimension)
        n_per_axis = max(1, n_axes // self.ndim)
        for axis in range(self.ndim):
            for d in range(self.ndim):
                if d == axis:
                    # Active dimension: sample broadly
                    x_real = torch.rand(n_per_axis, device=self.device) * 2 * r - r
                    x_imag = torch.rand(n_per_axis, device=self.device) * 2 * tp * imag_scale - tp * imag_scale
                else:
                    # Other dimensions: small values near zero
                    x_real = torch.randn(n_per_axis, device=self.device) * 0.3
                    x_imag = torch.randn(n_per_axis, device=self.device) * 0.1
                all_coords[d].append(torch.complex(x_real, x_imag))

        # Concatenate all regions
        coords = tuple(torch.cat(all_coords[d], dim=0) for d in range(self.ndim))

        return coords

    def save_checkpoint(self, path):
        """
        Save model checkpoint to disk.

        Parameters
        ----------
        path : str
            Path to save checkpoint file (.pt)
        """
        checkpoint = {
            'net_state_dict': self.net.state_dict(),
            'E': self.E.detach().cpu().numpy().item(),
            'quantum_numbers': self.quantum_numbers,
            'ndim': self.ndim,
            'hbar': self.hbar,
            'mass': self.mass,
            'loss_history': self.loss_history,
            'physics_loss_history': self.physics_loss_history,
            'curl_loss_history': self.curl_loss_history,
            'quant_loss_history': self.quant_loss_history,
            'supervision_loss_history': self.supervision_loss_history,
            'E_history': self.E_history,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """
        Load model checkpoint from disk.

        Parameters
        ----------
        path : str
            Path to checkpoint file (.pt)
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.E = torch.tensor(checkpoint['E'], device=self.device, requires_grad=True)
        self.loss_history = checkpoint.get('loss_history', [])
        self.physics_loss_history = checkpoint.get('physics_loss_history', [])
        self.curl_loss_history = checkpoint.get('curl_loss_history', [])
        self.quant_loss_history = checkpoint.get('quant_loss_history', [])
        self.supervision_loss_history = checkpoint.get('supervision_loss_history', [])
        self.E_history = checkpoint.get('E_history', [])

    def train(self, n_epochs=None, lr=None, n_collocation=None,
              physics_weight=1.0, curl_weight=0.5, quant_weight=1.0,
              supervision_weight=None, pole_reg_weight=0.01, quant_start_epoch=None,
              lr_decay_step=None, lr_decay_factor=0.5, verbose=True,
              checkpoint_path=None, checkpoint_interval=1800):
        """
        Train the network to find quantized energy and momentum field.

        Parameters
        ----------
        n_epochs : int or None
            Total training epochs. If None, uses dimension-aware default.
        lr : float or None
            Initial learning rate. If None, uses dimension-aware default.
        n_collocation : int or None
            Number of collocation points per batch. If None, uses dimension-aware default.
        physics_weight : float
            Weight for physics loss
        curl_weight : float
            Weight for curl loss
        quant_weight : float
            Weight for quantization loss
        supervision_weight : float or None
            Weight for supervision loss. If None, uses dimension-aware default.
        pole_reg_weight : float
            Weight for pole regularization
        quant_start_epoch : int or None
            Epoch to start applying quantization loss. If None, uses dimension-aware default.
        lr_decay_step : int or None
            Decay learning rate after this many epochs. If None, uses dimension-aware default.
        lr_decay_factor : float
            Factor to multiply learning rate by
        verbose : bool
            Print training progress
        checkpoint_path : str or None
            Path to save checkpoints. If None, no checkpoints are saved.
        checkpoint_interval : int
            Save checkpoint every N seconds (default: 1800 = 30 min)

        Returns
        -------
        dict
            Training result with energy, pole positions, and losses
        """
        # Dimension-aware defaults for training parameters
        # 3D: 15000 epochs, 900 collocation, lr=1e-3
        # 4D: 20000 epochs, 1200 collocation, lr=5e-4
        # 5D: 25000 epochs, 5000 collocation, lr=3.3e-4
        # 6D: 30000 epochs, 10000 collocation, lr=1e-4
        if n_epochs is None:
            if self.ndim >= 6:
                n_epochs = 30000
            else:
                n_epochs = 5000 * self.ndim
        if n_collocation is None:
            if self.ndim >= 6:
                n_collocation = 10000  # Much larger for 6D
            elif self.ndim >= 5:
                n_collocation = 5000
            else:
                n_collocation = 300 * self.ndim
        if lr is None:
            if self.ndim >= 6:
                lr = 1e-4  # Slower for stability in 6D
            else:
                lr = 1e-3 / max(1, self.ndim - 2)
        if supervision_weight is None:
            if self.ndim >= 6:
                supervision_weight = 20.0  # Stronger supervision in 6D
            else:
                supervision_weight = 2.5 * self.ndim
        if quant_start_epoch is None:
            quant_start_epoch = n_epochs // 5  # Start at 20% of total epochs
        if lr_decay_step is None:
            lr_decay_step = n_epochs // 3  # Decay at 33% of total epochs

        # Optimizer includes network params and energy
        all_params = list(self.net.parameters()) + [self.E]
        optimizer = torch.optim.Adam(all_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_decay_step, gamma=lr_decay_factor
        )

        self.loss_history = []
        self.physics_loss_history = []
        self.curl_loss_history = []
        self.quant_loss_history = []
        self.supervision_loss_history = []
        self.E_history = []

        # Checkpoint timing
        import time
        last_checkpoint_time = time.time()

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Sample collocation points
            coords_coll = self.sample_collocation_points(n_collocation)

            # Compute losses
            phys_loss = self.physics_loss(*coords_coll)
            curl_loss_val = self.curl_loss(*coords_coll)
            pole_reg = self.pole_regularization()
            sup_loss = self.supervision_loss()

            # Quantization loss only after initial training
            if epoch >= quant_start_epoch:
                quant_loss_val = self.quantization_loss()
            else:
                quant_loss_val = torch.tensor(0.0, device=self.device)

            # Total loss
            loss = (physics_weight * phys_loss +
                    curl_weight * curl_loss_val +
                    pole_reg_weight * pole_reg +
                    quant_weight * quant_loss_val +
                    supervision_weight * sup_loss)

            # Backpropagate
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Ensure energy stays positive
            with torch.no_grad():
                self.E.clamp_(min=0.1)

            # Track losses
            self.loss_history.append(loss.item())
            self.physics_loss_history.append(phys_loss.item())
            self.curl_loss_history.append(curl_loss_val.item())
            self.quant_loss_history.append(quant_loss_val.item())
            self.supervision_loss_history.append(sup_loss.item())
            self.E_history.append(self.E.item())

            if verbose and (epoch % 2000 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch:5d}: E={self.E.item():.4f}, "
                      f"phys={phys_loss.item():.2e}, curl={curl_loss_val.item():.2e}, "
                      f"quant={quant_loss_val.item():.2e}, sup={sup_loss.item():.2e}")

            # Save checkpoint periodically
            if checkpoint_path is not None:
                current_time = time.time()
                if current_time - last_checkpoint_time >= checkpoint_interval:
                    self.save_checkpoint(checkpoint_path)
                    if verbose:
                        print(f"  [Checkpoint saved to {checkpoint_path}]")
                    last_checkpoint_time = current_time

        # Save final checkpoint
        if checkpoint_path is not None:
            self.save_checkpoint(checkpoint_path)
            if verbose:
                print(f"[Final checkpoint saved to {checkpoint_path}]")

        # Final results
        poles = self.net.pole_positions()
        actions = []
        for d in range(self.ndim):
            J_d = self._compute_action(d).item()
            actions.append(J_d / self.hbar)

        return {
            'quantum_numbers': self.quantum_numbers,
            'E': self.E.item(),
            'J_over_hbar': actions,
            'poles': poles,
            'final_physics_loss': self.physics_loss_history[-1],
            'final_quant_loss': self.quant_loss_history[-1]
        }

    def evaluate(self, *coords):
        """
        Evaluate the learned momentum at positions.

        Parameters
        ----------
        coords : numpy.ndarray or torch.Tensor
            Complex coordinates for each dimension

        Returns
        -------
        tuple of numpy.ndarray
            Momentum values for each dimension
        """
        self.net.eval()

        # Convert to torch if needed
        coords_torch = []
        for x in coords:
            if isinstance(x, np.ndarray):
                coords_torch.append(
                    torch.from_numpy(x.astype(np.complex64)).to(self.device)
                )
            else:
                coords_torch.append(x.to(self.device))

        with torch.no_grad():
            momenta = self.net(*coords_torch)

        if isinstance(coords[0], np.ndarray):
            return tuple(p.cpu().numpy() for p in momenta)
        return momenta


class NonSepQuantumSolverND:
    """
    High-level solver for non-separable N-D quantum systems.

    Solves for individual quantum states by training a separate network
    for each target quantum number set.

    Parameters
    ----------
    potential : PotentialND
        N-dimensional potential function
    hbar : float
        Reduced Planck constant (default: 1.0)
    mass : float
        Particle mass (default: 1.0)
    device : str
        PyTorch device (default: auto-detect)
    """

    def __init__(self, potential, hbar=1.0, mass=1.0, device=None):
        self.potential = potential
        self.hbar = hbar
        self.mass = mass

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Cache for solved states
        self._solved_states = {}

    def solve_state(self, quantum_numbers, n_epochs=10000, verbose=True, **kwargs):
        """
        Solve for a specific quantum state.

        Parameters
        ----------
        quantum_numbers : tuple
            Target quantum numbers (n_0, n_1, ...)
        n_epochs : int
            Number of training epochs
        verbose : bool
            Print training progress
        **kwargs
            Additional arguments for QuantumNumberTrainerND.train()

        Returns
        -------
        dict
            Result dictionary with 'E', 'poles', etc.
        """
        trainer = QuantumNumberTrainerND(
            self.potential, quantum_numbers,
            hbar=self.hbar, mass=self.mass,
            device=self.device
        )

        result = trainer.train(n_epochs=n_epochs, verbose=verbose, **kwargs)

        # Cache result
        self._solved_states[tuple(quantum_numbers)] = result

        return result

    def solve_states(self, states_list, n_epochs=10000, verbose=True, **kwargs):
        """
        Solve for multiple quantum states.

        Parameters
        ----------
        states_list : list of tuple
            List of quantum number tuples
        n_epochs : int
            Number of training epochs per state
        verbose : bool
            Print progress
        **kwargs
            Additional arguments for training

        Returns
        -------
        list of dict
            Results for each state
        """
        results = []
        for qn in states_list:
            if verbose:
                print(f"\n=== Solving state {qn} ===")
            result = self.solve_state(qn, n_epochs=n_epochs, verbose=verbose, **kwargs)
            results.append(result)

        return results

    def exact_energy(self, quantum_numbers):
        """
        Get exact energy if potential supports it.

        Parameters
        ----------
        quantum_numbers : tuple
            Quantum numbers

        Returns
        -------
        float or None
            Exact energy or None if not available
        """
        if hasattr(self.potential, 'exact_energy'):
            return self.potential.exact_energy(quantum_numbers, self.hbar)
        return None

    def compare_with_dvr(self, dvr_energies, states=None):
        """
        Compare PINN energies with DVR reference.

        Parameters
        ----------
        dvr_energies : array-like
            DVR reference eigenvalues (sorted ascending)
        states : list of tuple or None
            Specific states to compare. If None, uses cached states.

        Returns
        -------
        dict
            Comparison results with errors
        """
        if states is not None:
            pinn_results = [self._solved_states.get(tuple(qn)) for qn in states
                          if tuple(qn) in self._solved_states]
        elif self._solved_states:
            pinn_results = list(self._solved_states.values())
        else:
            return None

        pinn_energies = sorted([r['E'] for r in pinn_results])
        dvr_energies = np.array(dvr_energies)

        n_compare = min(len(pinn_energies), len(dvr_energies))
        pinn_energies = np.array(pinn_energies[:n_compare])
        dvr_energies = dvr_energies[:n_compare]

        errors = np.abs(pinn_energies - dvr_energies)
        rel_errors = errors / np.abs(dvr_energies)

        return {
            'pinn_energies': list(pinn_energies),
            'dvr_energies': list(dvr_energies),
            'absolute_errors': list(errors),
            'relative_errors': list(rel_errors),
            'max_error': float(np.max(errors)),
            'mean_error': float(np.mean(errors)),
            'max_rel_error': float(np.max(rel_errors)),
            'mean_rel_error': float(np.mean(rel_errors))
        }
