"""
Quantum-number-conditioned PINN solver for 2D quantum Hamilton-Jacobi equation.

This module implements an improved PINN approach where the network architecture
is explicitly conditioned on target quantum numbers (n_x, n_y) rather than energy.
The key innovation is incorporating learnable pole positions that capture the
analytic structure of quantum momentum fields.

For state (n_x, n_y), the momentum field has n_x poles in the complex x-plane
and n_y poles in the complex y-plane. These poles are initialized at Hermite
polynomial zeros and trained alongside the network weights and energy.

The approach:
1. Build network ansatz with n_x poles in p_x and n_y poles in p_y
2. Train PINN with this pole structure
3. Find energy E where J_x = (n_x + 1/2)ℏ and J_y = (n_y + 1/2)ℏ

Reference:
    Leacock, R. A. & Padgett, M. J. (1983). "Hamilton-Jacobi/Action-Angle
    Quantum Mechanics." Phys. Rev. D 28, 2491-2502.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.special import roots_hermite
from scipy.optimize import brentq

from .potentials_2d import Potential2D, CoupledHarmonicOscillator, HarmonicOscillator2D


def hermite_zeros(n):
    """
    Compute zeros of the n-th Hermite polynomial H_n(x).

    These are the natural pole locations for the n-th excited state
    momentum field in a harmonic-like potential.

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
    # roots_hermite returns (zeros, weights) for Gauss-Hermite quadrature
    zeros, _ = roots_hermite(n)
    return np.sort(zeros)


class QuantumNumberPINN2D(nn.Module):
    """
    Neural network with learnable pole positions for state (n_x, n_y).

    The momentum field ansatz is:
        p_x = NN_x(x, y) + Σ_{k=1}^{n_x} 1/(x - x_k)
        p_y = NN_y(x, y) + Σ_{k=1}^{n_y} 1/(y - y_k)

    where x_k, y_k are LEARNABLE parameters initialized at Hermite zeros.

    Parameters
    ----------
    n_x : int
        Number of poles in x-direction (quantum number in x)
    n_y : int
        Number of poles in y-direction (quantum number in y)
    hidden_layers : int
        Number of hidden layers in smooth network (default: 4)
    hidden_size : int
        Neurons per hidden layer (default: 128)
    pole_scale : float
        Initial scale factor for pole positions (default: 1.0)
    """

    def __init__(self, n_x, n_y, hidden_layers=4, hidden_size=128, pole_scale=1.0):
        super().__init__()

        self.n_x = n_x
        self.n_y = n_y
        self.pole_scale = pole_scale

        # Learnable pole positions (initialized at scaled Hermite zeros)
        if n_x > 0:
            init_poles_x = hermite_zeros(n_x) * pole_scale
            self.poles_x = nn.Parameter(torch.tensor(init_poles_x, dtype=torch.float32))
        else:
            self.register_buffer('poles_x', torch.tensor([], dtype=torch.float32))

        if n_y > 0:
            init_poles_y = hermite_zeros(n_y) * pole_scale
            self.poles_y = nn.Parameter(torch.tensor(init_poles_y, dtype=torch.float32))
        else:
            self.register_buffer('poles_y', torch.tensor([], dtype=torch.float32))

        # Smooth part network: (Re(x), Im(x), Re(y), Im(y)) → (Re(p_x), Im(p_x), Re(p_y), Im(p_y))
        layers = []

        # Input layer
        layers.append(nn.Linear(4, hidden_size))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        # Output layer: 4 real outputs for complex (p_x, p_y)
        layers.append(nn.Linear(hidden_size, 4))

        self.smooth_net = nn.Sequential(*layers)

        # Initialize output layer to small values for stable training
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        with torch.no_grad():
            last_layer = self.smooth_net[-1]
            last_layer.weight.mul_(0.1)
            last_layer.bias.fill_(0.0)

    def forward(self, x_complex, y_complex):
        """
        Evaluate (p_x, p_y) for complex inputs (x, y).

        Parameters
        ----------
        x_complex : torch.Tensor
            Complex tensor of x-coordinates, shape (batch_size,)
        y_complex : torch.Tensor
            Complex tensor of y-coordinates, shape (batch_size,)

        Returns
        -------
        tuple of torch.Tensor
            (p_x, p_y) as complex tensors of shape (batch_size,)
        """
        # Split complex inputs into real and imaginary parts
        inputs = torch.stack([
            x_complex.real, x_complex.imag,
            y_complex.real, y_complex.imag,
        ], dim=-1)

        # Get smooth part from network
        output = self.smooth_net(inputs)

        # Extract smooth components
        p_x_smooth = torch.complex(output[..., 0], output[..., 1])
        p_y_smooth = torch.complex(output[..., 2], output[..., 3])

        # Add pole contributions to p_x
        p_x = p_x_smooth
        for k in range(self.n_x):
            # Pole at x = poles_x[k] (real position, extends to complex plane)
            pole_pos = self.poles_x[k].to(x_complex.dtype)
            p_x = p_x + 1.0 / (x_complex - pole_pos)

        # Add pole contributions to p_y
        p_y = p_y_smooth
        for k in range(self.n_y):
            pole_pos = self.poles_y[k].to(y_complex.dtype)
            p_y = p_y + 1.0 / (y_complex - pole_pos)

        return p_x, p_y

    def pole_positions(self):
        """
        Return current pole positions.

        Returns
        -------
        dict
            Dictionary with 'x' and 'y' keys containing numpy arrays of poles
        """
        poles_x = self.poles_x.detach().cpu().numpy() if self.n_x > 0 else np.array([])
        poles_y = self.poles_y.detach().cpu().numpy() if self.n_y > 0 else np.array([])
        return {'x': poles_x, 'y': poles_y}


class QuantumNumberTrainer:
    """
    Train PINN for specific quantum numbers (n_x, n_y) and find quantized energy.

    This trainer jointly optimizes:
    - Network weights (smooth part of momentum)
    - Pole positions (shifted from Hermite zeros by coupling)
    - Energy E (driven to value satisfying quantization conditions)

    Parameters
    ----------
    potential : Potential2D
        2D potential function V(x, y)
    n_x : int
        Target quantum number in x
    n_y : int
        Target quantum number in y
    hbar : float
        Reduced Planck constant (default: 1.0)
    mass : float
        Particle mass (default: 1.0)
    hidden_layers : int
        Network hidden layers (default: 4)
    hidden_size : int
        Neurons per hidden layer (default: 128)
    device : str
        PyTorch device ('cpu' or 'cuda')
    """

    def __init__(self, potential, n_x, n_y, hbar=1.0, mass=1.0,
                 hidden_layers=4, hidden_size=128, device=None):
        self.potential = potential
        self.n_x = n_x
        self.n_y = n_y
        self.hbar = hbar
        self.mass = mass

        # Auto-detect GPU
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Estimate initial energy from uncoupled harmonic approximation
        E_init = float((n_x + n_y + 1) * hbar)

        # Estimate pole scale from energy
        pole_scale = np.sqrt(E_init) if E_init > 0 else 1.0

        # Create network
        self.net = QuantumNumberPINN2D(
            n_x, n_y,
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
        self.E_history = []

    def _potential_torch(self, x, y):
        """Evaluate potential at complex (x, y) using PyTorch operations."""
        if isinstance(self.potential, CoupledHarmonicOscillator):
            return 0.5 * (x**2 + y**2) + self.potential.coupling * x * y
        elif isinstance(self.potential, HarmonicOscillator2D):
            return (0.5 * self.potential.mass * self.potential.omega_x**2 * x**2 +
                    0.5 * self.potential.mass * self.potential.omega_y**2 * y**2)
        else:
            # Generic: assume harmonic base
            return 0.5 * (x**2 + y**2)

    def physics_loss(self, x_coll, y_coll):
        """
        Compute physics loss enforcing the 2D quantum HJ equation.

        L = mean(|p_x² + p_y² + iℏ(∂p_x/∂x + ∂p_y/∂y) - 2m(E - V)|²)

        Parameters
        ----------
        x_coll, y_coll : torch.Tensor
            Complex collocation points

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        h = 1e-4

        # Evaluate network at collocation points
        p_x, p_y = self.net(x_coll, y_coll)

        # Compute ∂p_x/∂x using finite differences
        p_x_plus, _ = self.net(x_coll + h, y_coll)
        p_x_minus, _ = self.net(x_coll - h, y_coll)
        dp_x_dx = (p_x_plus - p_x_minus) / (2 * h)

        # Compute ∂p_y/∂y using finite differences
        _, p_y_plus = self.net(x_coll, y_coll + h)
        _, p_y_minus = self.net(x_coll, y_coll - h)
        dp_y_dy = (p_y_plus - p_y_minus) / (2 * h)

        # Evaluate potential (convert to complex type)
        V = self._potential_torch(x_coll, y_coll)

        # Energy as complex for computation
        E_complex = self.E.to(x_coll.dtype)

        # QHJ residual: p_x² + p_y² + iℏ(∂p_x/∂x + ∂p_y/∂y) - 2m(E - V)
        residual = (p_x**2 + p_y**2
                   + 1j * self.hbar * (dp_x_dx + dp_y_dy)
                   - 2 * self.mass * (E_complex - V))

        # Loss: mean of squared magnitude
        loss = torch.mean(torch.abs(residual)**2)

        return loss

    def curl_loss(self, x_coll, y_coll):
        """
        Compute curl loss enforcing irrotationality.

        For p to be a gradient field: ∂p_x/∂y = ∂p_y/∂x

        Parameters
        ----------
        x_coll, y_coll : torch.Tensor
            Complex collocation points

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        h = 1e-4

        # Compute ∂p_x/∂y
        p_x_y_plus, _ = self.net(x_coll, y_coll + h)
        p_x_y_minus, _ = self.net(x_coll, y_coll - h)
        dp_x_dy = (p_x_y_plus - p_x_y_minus) / (2 * h)

        # Compute ∂p_y/∂x
        _, p_y_x_plus = self.net(x_coll + h, y_coll)
        _, p_y_x_minus = self.net(x_coll - h, y_coll)
        dp_y_dx = (p_y_x_plus - p_y_x_minus) / (2 * h)

        # Curl residual (should be zero for gradient field)
        curl = dp_x_dy - dp_y_dx

        loss = torch.mean(torch.abs(curl)**2)

        return loss

    def pole_regularization(self):
        """
        Regularization to keep poles in a reasonable region.

        Penalizes poles that drift too far from their initial positions.

        Returns
        -------
        torch.Tensor
            Regularization loss
        """
        loss = torch.tensor(0.0, device=self.device)

        # Regularize pole positions relative to expected scale
        E_val = max(self.E.item(), 0.5)
        expected_scale = np.sqrt(2 * E_val)

        if self.n_x > 0:
            # Penalize poles outside classical region
            pole_sq = self.net.poles_x ** 2
            loss = loss + torch.sum(torch.relu(pole_sq - expected_scale**2))

        if self.n_y > 0:
            pole_sq = self.net.poles_y ** 2
            loss = loss + torch.sum(torch.relu(pole_sq - expected_scale**2))

        return loss * 0.01  # Small regularization weight

    def supervision_loss(self, n_points=100):
        """
        Supervision loss for momentum structure.

        For harmonic-like potentials, the exact quantum momentum is:
            p_x = -i * x + sum(1/(x - x_k))  for state n_x
            p_y = -i * y + sum(1/(y - y_k))  for state n_y

        where x_k, y_k are zeros of Hermite polynomials (the pole locations).

        The network already includes explicit poles, so it needs to learn
        the smooth part: -i * x for p_x and -i * y for p_y.

        Parameters
        ----------
        n_points : int
            Number of supervision points

        Returns
        -------
        torch.Tensor
            Supervision loss
        """
        if not isinstance(self.potential, (HarmonicOscillator2D, CoupledHarmonicOscillator)):
            return torch.tensor(0.0, device=self.device)

        # Sample points in complex plane, avoiding poles
        n_pts = int(np.sqrt(n_points))

        # For excited states, avoid sampling near poles (Hermite zeros)
        pole_positions_x = self.net.poles_x.detach().cpu().numpy() if self.n_x > 0 else []
        pole_positions_y = self.net.poles_y.detach().cpu().numpy() if self.n_y > 0 else []

        # Create x grid avoiding poles
        if self.n_x > 0:
            # Avoid regions near poles
            x_r_left = torch.linspace(-3.0, -0.3, n_pts // 2, device=self.device)
            x_r_right = torch.linspace(0.3, 3.0, n_pts // 2, device=self.device)
            x_r = torch.cat([x_r_left, x_r_right])
        else:
            x_r = torch.linspace(-3.0, 3.0, n_pts, device=self.device)

        x_i = torch.linspace(-1.5, 1.5, n_pts, device=self.device)
        xr_grid, xi_grid = torch.meshgrid(x_r, x_i, indexing='ij')
        x_for_px = torch.complex(xr_grid.flatten(), xi_grid.flatten())
        y_zero = torch.zeros_like(x_for_px)

        # Create y grid avoiding poles
        if self.n_y > 0:
            y_r_left = torch.linspace(-3.0, -0.3, n_pts // 2, device=self.device)
            y_r_right = torch.linspace(0.3, 3.0, n_pts // 2, device=self.device)
            y_r = torch.cat([y_r_left, y_r_right])
        else:
            y_r = torch.linspace(-3.0, 3.0, n_pts, device=self.device)

        y_i = torch.linspace(-1.5, 1.5, n_pts, device=self.device)
        yr_grid, yi_grid = torch.meshgrid(y_r, y_i, indexing='ij')
        y_for_py = torch.complex(yr_grid.flatten(), yi_grid.flatten())
        x_zero = torch.zeros_like(y_for_py)

        # Supervise the SMOOTH part of the network output
        # For UNCOUPLED oscillator: p_x = -i*x, p_y = -i*y
        # For COUPLED oscillator with V = (x² + y²)/2 + λxy:
        #   The normal mode frequencies are ω± = √(1 ± λ)
        #   The exact momentum involves cross-terms:
        #   p_x = -i * (α_xx * x + α_xy * y)
        #   p_y = -i * (α_xy * x + α_xx * y)
        #   where α_xx = (ω+ + ω-)/2, α_xy = (ω+ - ω-)/2

        if isinstance(self.potential, CoupledHarmonicOscillator):
            omega_plus = self.potential.omega_plus
            omega_minus = self.potential.omega_minus
            alpha_xx = (omega_plus + omega_minus) / 2
            alpha_xy = (omega_plus - omega_minus) / 2
        else:
            alpha_xx = 1.0
            alpha_xy = 0.0

        # Expected smooth parts with coupling
        # Note: for supervision at y=0, cross term vanishes for p_x
        # But we need the full formula for general points
        p_x_smooth_expected = -1j * (alpha_xx * x_for_px)  # y=0 for these points
        p_y_smooth_expected = -1j * (alpha_xx * y_for_py)  # x=0 for these points

        # Get the smooth part from network (before adding poles)
        # We need to evaluate the smooth_net directly
        x_input = torch.stack([
            x_for_px.real, x_for_px.imag,
            y_zero.real, y_zero.imag
        ], dim=-1)
        y_input = torch.stack([
            x_zero.real, x_zero.imag,
            y_for_py.real, y_for_py.imag
        ], dim=-1)

        smooth_out_x = self.net.smooth_net(x_input)
        p_x_smooth_actual = torch.complex(smooth_out_x[..., 0], smooth_out_x[..., 1])

        smooth_out_y = self.net.smooth_net(y_input)
        p_y_smooth_actual = torch.complex(smooth_out_y[..., 2], smooth_out_y[..., 3])

        # Loss: match smooth parts along axes
        loss_x = torch.mean(torch.abs(p_x_smooth_actual - p_x_smooth_expected)**2)
        loss_y = torch.mean(torch.abs(p_y_smooth_actual - p_y_smooth_expected)**2)

        # For coupled oscillator, also supervise at 2D points to capture cross-terms
        if isinstance(self.potential, CoupledHarmonicOscillator) and alpha_xy != 0:
            n_2d = n_pts // 2
            x_2d = torch.linspace(-2.0, 2.0, n_2d, device=self.device)
            y_2d = torch.linspace(-2.0, 2.0, n_2d, device=self.device)
            x_grid, y_grid = torch.meshgrid(x_2d, y_2d, indexing='ij')
            x_2d_pts = torch.complex(x_grid.flatten(), torch.zeros_like(x_grid.flatten()))
            y_2d_pts = torch.complex(y_grid.flatten(), torch.zeros_like(y_grid.flatten()))

            # Expected smooth parts with cross-coupling
            # p_x = -i * (alpha_xx * x + alpha_xy * y)
            # p_y = -i * (alpha_xy * x + alpha_xx * y)
            p_x_smooth_expected_2d = -1j * (alpha_xx * x_2d_pts + alpha_xy * y_2d_pts)
            p_y_smooth_expected_2d = -1j * (alpha_xy * x_2d_pts + alpha_xx * y_2d_pts)

            # Get network smooth output at 2D points
            xy_input = torch.stack([
                x_2d_pts.real, x_2d_pts.imag,
                y_2d_pts.real, y_2d_pts.imag
            ], dim=-1)
            smooth_out_2d = self.net.smooth_net(xy_input)
            p_x_smooth_actual_2d = torch.complex(smooth_out_2d[..., 0], smooth_out_2d[..., 1])
            p_y_smooth_actual_2d = torch.complex(smooth_out_2d[..., 2], smooth_out_2d[..., 3])

            loss_2d = (torch.mean(torch.abs(p_x_smooth_actual_2d - p_x_smooth_expected_2d)**2) +
                       torch.mean(torch.abs(p_y_smooth_actual_2d - p_y_smooth_expected_2d)**2))

            return loss_x + loss_y + loss_2d

        return loss_x + loss_y

    def quantization_loss(self, n_points=50):
        """
        Loss driving energy to satisfy quantization conditions.

        Computes action integrals and penalizes deviation from
        J_x = (n_x + 0.5)ℏ and J_y = (n_y + 0.5)ℏ.

        Parameters
        ----------
        n_points : int
            Number of points for action integral

        Returns
        -------
        torch.Tensor
            Quantization loss
        """
        # Target actions
        target_x = (self.n_x + 0.5) * self.hbar
        target_y = (self.n_y + 0.5) * self.hbar

        # Compute action integrals using contour integration
        # Use allow_grad=True so gradients flow to network and energy
        J_x = self._compute_action_x(n_points, allow_grad=True)
        J_y = self._compute_action_y(n_points, allow_grad=True)

        # Quantization loss
        loss = (J_x - target_x)**2 + (J_y - target_y)**2

        return loss

    def _compute_action_x(self, n_points=50, allow_grad=False):
        """
        Compute x-direction action integral J_x using contour integration.

        Creates elliptical contour in complex x-plane, avoiding learned poles.

        Parameters
        ----------
        n_points : int
            Number of contour points
        allow_grad : bool
            If True, allow gradients to flow for training

        Returns
        -------
        torch.Tensor
            Action integral J_x (real scalar)
        """
        E_val = max(self.E.item(), 0.1)
        y_fixed = 0.0

        # Estimate turning points
        tp_estimate = np.sqrt(2 * E_val)

        # Create elliptical contour
        center = 0.0
        a = 1.3 * tp_estimate  # Semi-major axis (real)
        b = 0.4 * tp_estimate  # Semi-minor axis (imaginary)

        theta = torch.linspace(0, 2 * np.pi, n_points, device=self.device, dtype=torch.float32)
        x_real = center + a * torch.cos(theta)
        x_imag = b * torch.sin(theta)

        # Ensure contour avoids poles - offset imaginary part if needed
        if self.n_x > 0:
            poles = self.net.poles_x.detach()
            min_clearance = 0.2 * b
            # Check if any pole is within the contour
            for pole in poles:
                pole_val = pole.item()
                if abs(pole_val - center) < a:
                    # Pole is within horizontal extent - ensure imaginary clearance
                    b = max(b, min_clearance + 0.1)

        x_contour = torch.complex(x_real, x_imag)
        y_contour = torch.full_like(x_contour, y_fixed + 0j)

        # Evaluate momentum on contour
        if allow_grad:
            p_x, _ = self.net(x_contour, y_contour)
        else:
            with torch.no_grad():
                p_x, _ = self.net(x_contour, y_contour)

        # Trapezoidal integration: ∮ p_x dx
        integral = torch.tensor(0j, dtype=torch.complex64, device=self.device)
        for i in range(n_points):
            dx = x_contour[(i + 1) % n_points] - x_contour[i]
            p_avg = (p_x[i] + p_x[(i + 1) % n_points]) / 2
            integral = integral + p_avg * dx

        # J_x = (1/2πi) ∮ p_x dx
        J_x = integral / (2 * np.pi * 1j)

        # Return real part (action is real)
        return J_x.real

    def _compute_action_y(self, n_points=50, allow_grad=False):
        """
        Compute y-direction action integral J_y using contour integration.

        Parameters
        ----------
        n_points : int
            Number of contour points
        allow_grad : bool
            If True, allow gradients to flow for training

        Returns
        -------
        torch.Tensor
            Action integral J_y (real scalar)
        """
        E_val = max(self.E.item(), 0.1)
        x_fixed = 0.0

        # Estimate turning points
        tp_estimate = np.sqrt(2 * E_val)

        # Create elliptical contour
        center = 0.0
        a = 1.3 * tp_estimate
        b = 0.4 * tp_estimate

        theta = torch.linspace(0, 2 * np.pi, n_points, device=self.device, dtype=torch.float32)
        y_real = center + a * torch.cos(theta)
        y_imag = b * torch.sin(theta)

        y_contour = torch.complex(y_real, y_imag)
        x_contour = torch.full_like(y_contour, x_fixed + 0j)

        # Evaluate momentum on contour
        if allow_grad:
            _, p_y = self.net(x_contour, y_contour)
        else:
            with torch.no_grad():
                _, p_y = self.net(x_contour, y_contour)

        # Trapezoidal integration
        integral = torch.tensor(0j, dtype=torch.complex64, device=self.device)
        for i in range(n_points):
            dy = y_contour[(i + 1) % n_points] - y_contour[i]
            p_avg = (p_y[i] + p_y[(i + 1) % n_points]) / 2
            integral = integral + p_avg * dy

        J_y = integral / (2 * np.pi * 1j)

        return J_y.real

    def sample_collocation_points(self, n_points, imag_scale=0.3):
        """
        Sample collocation points in the 2D complex plane.

        Samples avoid pole positions for numerical stability.

        Parameters
        ----------
        n_points : int
            Number of points to sample
        imag_scale : float
            Scale for imaginary parts relative to turning point

        Returns
        -------
        tuple of torch.Tensor
            (x_complex, y_complex) collocation points
        """
        E_val = max(self.E.item(), 0.5)
        tp = np.sqrt(2 * E_val)
        r = tp * 2.5

        # Real parts
        x_real = torch.rand(n_points, device=self.device) * 2 * r - r
        y_real = torch.rand(n_points, device=self.device) * 2 * r - r

        # Imaginary parts (smaller for stability)
        x_imag = torch.rand(n_points, device=self.device) * 2 * tp * imag_scale - tp * imag_scale
        y_imag = torch.rand(n_points, device=self.device) * 2 * tp * imag_scale - tp * imag_scale

        x = torch.complex(x_real, x_imag)
        y = torch.complex(y_real, y_imag)

        return x, y

    def train(self, n_epochs=10000, lr=1e-3, n_collocation=1000,
              physics_weight=1.0, curl_weight=0.5, quant_weight=1.0,
              supervision_weight=1.0, pole_reg_weight=0.01, quant_start_epoch=2000,
              lr_decay_step=3000, lr_decay_factor=0.5, verbose=True):
        """
        Train the network to find quantized energy and momentum field.

        Training has two phases:
        1. Physics-only phase: Learn momentum field structure
        2. Quantization phase: Drive energy to correct value

        Parameters
        ----------
        n_epochs : int
            Total training epochs
        lr : float
            Initial learning rate
        n_collocation : int
            Number of collocation points per batch
        physics_weight : float
            Weight for physics loss
        curl_weight : float
            Weight for curl loss
        quant_weight : float
            Weight for quantization loss
        supervision_weight : float
            Weight for supervision loss (for ground state)
        pole_reg_weight : float
            Weight for pole regularization
        quant_start_epoch : int
            Epoch to start applying quantization loss
        lr_decay_step : int
            Decay learning rate after this many epochs
        lr_decay_factor : float
            Factor to multiply learning rate by
        verbose : bool
            Print training progress

        Returns
        -------
        dict
            Training result with energy, pole positions, and losses
        """
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

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Sample collocation points
            x_coll, y_coll = self.sample_collocation_points(n_collocation)

            # Compute losses
            phys_loss = self.physics_loss(x_coll, y_coll)
            curl_loss_val = self.curl_loss(x_coll, y_coll)
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

            # Gradient clipping for stability
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

        # Final results
        poles = self.net.pole_positions()
        J_x = self._compute_action_x().item()
        J_y = self._compute_action_y().item()

        return {
            'n_x': self.n_x,
            'n_y': self.n_y,
            'E': self.E.item(),
            'J_x_over_hbar': J_x / self.hbar,
            'J_y_over_hbar': J_y / self.hbar,
            'poles_x': poles['x'],
            'poles_y': poles['y'],
            'final_physics_loss': self.physics_loss_history[-1],
            'final_quant_loss': self.quant_loss_history[-1]
        }

    def evaluate(self, x, y):
        """
        Evaluate the learned momentum (p_x, p_y) at positions (x, y).

        Parameters
        ----------
        x, y : numpy.ndarray or torch.Tensor
            Complex coordinates

        Returns
        -------
        tuple of numpy.ndarray
            (p_x, p_y) complex momentum values
        """
        self.net.eval()

        # Convert to torch if needed
        if isinstance(x, np.ndarray):
            x_torch = torch.from_numpy(x.astype(np.complex64)).to(self.device)
            y_torch = torch.from_numpy(y.astype(np.complex64)).to(self.device)
        else:
            x_torch = x.to(self.device)
            y_torch = y.to(self.device)

        with torch.no_grad():
            p_x, p_y = self.net(x_torch, y_torch)

        if isinstance(x, np.ndarray):
            return p_x.cpu().numpy(), p_y.cpu().numpy()
        return p_x, p_y


class NonSepQuantumSolver:
    """
    High-level solver for non-separable 2D quantum systems.

    Solves for individual quantum states by training a separate network
    for each target (n_x, n_y).

    Parameters
    ----------
    potential : Potential2D
        2D potential function V(x, y)
    hbar : float
        Reduced Planck constant (default: 1.0)
    mass : float
        Particle mass (default: 1.0)
    device : str
        PyTorch device (default: auto-detect)

    Example
    -------
    >>> from quantum_hj import CoupledHarmonicOscillator
    >>> pot = CoupledHarmonicOscillator(coupling=0.2)
    >>> solver = NonSepQuantumSolver(pot)
    >>> result = solver.solve_state(1, 0)  # First excited state in x
    >>> print(f"Energy: {result['E']:.4f}")
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

    def solve_state(self, n_x, n_y, n_epochs=10000, verbose=True, **kwargs):
        """
        Solve for a specific quantum state (n_x, n_y).

        Parameters
        ----------
        n_x : int
            Quantum number in x-direction
        n_y : int
            Quantum number in y-direction
        n_epochs : int
            Number of training epochs
        verbose : bool
            Print training progress
        **kwargs
            Additional arguments for QuantumNumberTrainer.train()

        Returns
        -------
        dict
            Result dictionary with 'E', 'poles_x', 'poles_y', etc.
        """
        trainer = QuantumNumberTrainer(
            self.potential, n_x, n_y,
            hbar=self.hbar, mass=self.mass,
            device=self.device
        )

        result = trainer.train(n_epochs=n_epochs, verbose=verbose, **kwargs)

        # Cache result
        self._solved_states[(n_x, n_y)] = result

        return result

    def solve_states(self, states, n_epochs=10000, verbose=True, **kwargs):
        """
        Solve for multiple quantum states.

        Parameters
        ----------
        states : list of tuple
            List of (n_x, n_y) tuples
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
        for n_x, n_y in states:
            if verbose:
                print(f"\n=== Solving state ({n_x}, {n_y}) ===")
            result = self.solve_state(n_x, n_y, n_epochs=n_epochs, verbose=verbose, **kwargs)
            results.append(result)

        return results

    def solve_up_to(self, n_max, n_epochs=10000, verbose=True, **kwargs):
        """
        Solve for all states with n_x, n_y <= n_max.

        Parameters
        ----------
        n_max : int
            Maximum quantum number in each direction
        n_epochs : int
            Number of training epochs per state
        verbose : bool
            Print progress
        **kwargs
            Additional arguments for training

        Returns
        -------
        list of dict
            Results sorted by energy
        """
        states = [(n_x, n_y) for n_x in range(n_max + 1)
                  for n_y in range(n_max + 1)]

        results = self.solve_states(states, n_epochs=n_epochs, verbose=verbose, **kwargs)

        # Sort by energy
        results.sort(key=lambda r: r['E'])

        return results

    def compare_with_dvr(self, dvr_energies, states=None, n_max=2):
        """
        Compare PINN energies with DVR reference.

        Parameters
        ----------
        dvr_energies : array-like
            DVR reference eigenvalues (sorted ascending)
        states : list of tuple or None
            Specific states to compare. If None, uses cached states.
        n_max : int
            If states is None and no cached states, solve up to n_max

        Returns
        -------
        dict
            Comparison results with errors
        """
        # Get PINN results
        if states is not None:
            pinn_results = [self._solved_states.get((nx, ny)) for nx, ny in states
                          if (nx, ny) in self._solved_states]
        elif self._solved_states:
            pinn_results = list(self._solved_states.values())
        else:
            pinn_results = self.solve_up_to(n_max)

        pinn_energies = sorted([r['E'] for r in pinn_results])
        dvr_energies = np.array(dvr_energies)

        # Match by number of states
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

    def exact_energy(self, n_x, n_y):
        """
        Get exact energy if potential supports it.

        Parameters
        ----------
        n_x, n_y : int
            Quantum numbers

        Returns
        -------
        float or None
            Exact energy or None if not available
        """
        if isinstance(self.potential, CoupledHarmonicOscillator):
            # For coupled oscillator, (n_x, n_y) maps to normal modes
            # The mapping depends on the specific state
            return self.potential.exact_energy(n_x, n_y, self.hbar)
        elif isinstance(self.potential, HarmonicOscillator2D):
            return self.potential.exact_energy(n_x, n_y, self.hbar)
        return None
