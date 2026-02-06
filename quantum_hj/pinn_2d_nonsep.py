"""
Energy-conditioned PINN solver for non-separable 2D quantum Hamilton-Jacobi equation.

This module extends the 2D PINN approach to handle non-separable potentials
where V(x,y) ≠ V_x(x) + V_y(y). The key innovation is conditioning the network
on energy E, allowing a single trained network to approximate solutions across
a range of energies.

For non-separable potentials, the quantum momenta p_x and p_y both depend on
both coordinates (x, y), unlike the separable case where p_x = p_x(x) and
p_y = p_y(y).

The EBK (Einstein-Brillouin-Keller) quantization conditions are used to find
bound state energies:
    J_x(E) = ℏ(n_x + 1/2)
    J_y(E) = ℏ(n_y + 1/2)

Reference:
    Leacock, R. A. & Padgett, M. J. (1983). "Hamilton-Jacobi/Action-Angle
    Quantum Mechanics." Phys. Rev. D 28, 2491-2502.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import brentq, minimize_scalar
from scipy.integrate import quad

from .potentials_2d import Potential2D, CoupledHarmonicOscillator, HenonHeiles, HarmonicOscillator2D


class EnergyConditionedMLP(nn.Module):
    """
    Neural network for energy-conditioned 2D complex momentum (p_x, p_y)(x, y, E).

    Takes complex (x, y) and complex E as input (6 real channels) and outputs
    complex (p_x, p_y) (4 real channels). The network learns to approximate
    the quantum momentum field as a function of position and energy.

    Parameters
    ----------
    hidden_layers : int
        Number of hidden layers (default: 5)
    hidden_size : int
        Number of neurons per hidden layer (default: 256)
    """

    def __init__(self, hidden_layers=5, hidden_size=256):
        super().__init__()

        layers = []

        # Input layer: (Re(x), Im(x), Re(y), Im(y), Re(E), Im(E)) -> hidden_size
        layers.append(nn.Linear(6, hidden_size))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        # Output layer: hidden_size -> (Re(p_x), Im(p_x), Re(p_y), Im(p_y))
        layers.append(nn.Linear(hidden_size, 4))

        self.net = nn.Sequential(*layers)

        # Initialize output layer to small values
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        with torch.no_grad():
            last_layer = self.net[-1]
            last_layer.weight.mul_(0.1)
            last_layer.bias.fill_(0.0)

    def forward(self, x_complex, y_complex, E_complex):
        """
        Evaluate (p_x, p_y) for complex inputs (x, y, E).

        Parameters
        ----------
        x_complex : torch.Tensor
            Complex tensor of x-coordinates, shape (batch_size,)
        y_complex : torch.Tensor
            Complex tensor of y-coordinates, shape (batch_size,)
        E_complex : torch.Tensor
            Complex tensor of energy values, shape (batch_size,)

        Returns
        -------
        tuple of torch.Tensor
            (p_x, p_y) as complex tensors of shape (batch_size,)
        """
        # Split complex inputs into real and imaginary parts
        inputs = torch.stack([
            x_complex.real, x_complex.imag,
            y_complex.real, y_complex.imag,
            E_complex.real, E_complex.imag
        ], dim=-1)

        # Forward through network
        output = self.net(inputs)

        # Extract components
        p_x = torch.complex(output[..., 0], output[..., 1])
        p_y = torch.complex(output[..., 2], output[..., 3])

        return p_x, p_y


class QuantumHJPINN2DNonSep:
    """
    PINN trainer for non-separable 2D quantum Hamilton-Jacobi equation.

    Trains a neural network to learn (p_x, p_y) conditioned on energy E
    by minimizing:
    - Physics loss: QHJ equation residual
    - Curl loss: Irrotationality constraint (p must be a gradient)
    - Asymptotic loss: Correct behavior in classically forbidden region

    Parameters
    ----------
    potential : Potential2D
        2D potential function V(x, y)
    E_range : tuple
        (E_min, E_max) energy range for training
    hbar : float
        Reduced Planck constant (default: 1.0)
    mass : float
        Particle mass (default: 1.0)
    hidden_layers : int
        Number of hidden layers (default: 5)
    hidden_size : int
        Neurons per hidden layer (default: 256)
    device : str
        PyTorch device ('cpu' or 'cuda')
    """

    def __init__(self, potential, E_range, hbar=1.0, mass=1.0,
                 hidden_layers=5, hidden_size=256, device=None):
        self.potential = potential
        self.E_min, self.E_max = E_range
        self.hbar = hbar
        self.mass = mass

        # Auto-detect GPU if device not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Create network
        self.net = EnergyConditionedMLP(hidden_layers, hidden_size).to(self.device)

        # Training history
        self.loss_history = []
        self.physics_loss_history = []
        self.curl_loss_history = []
        self.asymptotic_loss_history = []

    def _potential_torch(self, x, y):
        """Evaluate potential at complex (x, y) using PyTorch operations."""
        if isinstance(self.potential, CoupledHarmonicOscillator):
            return 0.5 * (x**2 + y**2) + self.potential.coupling * x * y
        elif isinstance(self.potential, HenonHeiles):
            return (0.5 * (x**2 + y**2) +
                    self.potential.coupling * (x**2 * y - y**3 / 3.0))
        else:
            # Generic: assume potential is evaluable
            return 0.5 * (x**2 + y**2)

    def physics_loss(self, x_coll, y_coll, E_coll):
        """
        Compute physics loss enforcing the 2D quantum HJ equation.

        L = mean(|p_x² + p_y² + iℏ(∂p_x/∂x + ∂p_y/∂y) - 2m(E - V)|²)

        Parameters
        ----------
        x_coll, y_coll, E_coll : torch.Tensor
            Complex collocation points

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        h = 1e-4

        # Evaluate network at collocation points
        p_x, p_y = self.net(x_coll, y_coll, E_coll)

        # Compute ∂p_x/∂x using finite differences
        p_x_plus, _ = self.net(x_coll + h, y_coll, E_coll)
        p_x_minus, _ = self.net(x_coll - h, y_coll, E_coll)
        dp_x_dx = (p_x_plus - p_x_minus) / (2 * h)

        # Compute ∂p_y/∂y using finite differences
        _, p_y_plus = self.net(x_coll, y_coll + h, E_coll)
        _, p_y_minus = self.net(x_coll, y_coll - h, E_coll)
        dp_y_dy = (p_y_plus - p_y_minus) / (2 * h)

        # Evaluate potential
        V = self._potential_torch(x_coll, y_coll)

        # QHJ residual: p_x² + p_y² + iℏ(∂p_x/∂x + ∂p_y/∂y) - 2m(E - V)
        residual = (p_x**2 + p_y**2
                   + 1j * self.hbar * (dp_x_dx + dp_y_dy)
                   - 2 * self.mass * (E_coll - V))

        # Loss: mean of squared magnitude
        loss = torch.mean(torch.abs(residual)**2)

        return loss

    def curl_loss(self, x_coll, y_coll, E_coll):
        """
        Compute curl loss enforcing irrotationality.

        For p to be a gradient field: ∂p_x/∂y = ∂p_y/∂x

        Parameters
        ----------
        x_coll, y_coll, E_coll : torch.Tensor
            Complex collocation points

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        h = 1e-4

        # Compute ∂p_x/∂y
        p_x_y_plus, _ = self.net(x_coll, y_coll + h, E_coll)
        p_x_y_minus, _ = self.net(x_coll, y_coll - h, E_coll)
        dp_x_dy = (p_x_y_plus - p_x_y_minus) / (2 * h)

        # Compute ∂p_y/∂x
        _, p_y_x_plus = self.net(x_coll + h, y_coll, E_coll)
        _, p_y_x_minus = self.net(x_coll - h, y_coll, E_coll)
        dp_y_dx = (p_y_x_plus - p_y_x_minus) / (2 * h)

        # Curl residual
        curl = dp_x_dy - dp_y_dx

        loss = torch.mean(torch.abs(curl)**2)

        return loss

    def asymptotic_loss(self, x_coll, y_coll, E_coll):
        """
        Enforce correct asymptotic behavior in classically forbidden region.

        Deep in the forbidden region (V >> E), the quantum momentum should be
        approximately imaginary with p ≈ ±i√(2m(V-E)).

        Parameters
        ----------
        x_coll, y_coll, E_coll : torch.Tensor
            Complex collocation points (real part only used here)

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        # Sample points in forbidden region
        n_asymp = min(100, len(x_coll))

        # Take real parts and sample far from origin
        r = 3.0  # Sample at large radius
        theta = torch.linspace(0, 2*np.pi, n_asymp, device=self.device)
        x_far = r * torch.cos(theta) + 0j
        y_far = r * torch.sin(theta) + 0j

        # Use lowest energy in range
        E_low = torch.full((n_asymp,), self.E_min + 0j, device=self.device)

        # Get momentum
        p_x, p_y = self.net(x_far, y_far, E_low)

        # In forbidden region, expect |p|² ≈ -2m(E - V) < 0
        # This means p should be primarily imaginary
        V = self._potential_torch(x_far.real, y_far.real)

        # For V > E, p should be imaginary: Im(p)² > Re(p)²
        # We encourage Im(p_x)² + Im(p_y)² ≈ 2m(V - E)
        expected_p_sq = 2 * self.mass * (V.real - self.E_min)
        mask = expected_p_sq > 0

        if mask.sum() > 0:
            actual_im_sq = p_x[mask].imag**2 + p_y[mask].imag**2
            loss = torch.mean((actual_im_sq - expected_p_sq[mask])**2)
        else:
            loss = torch.tensor(0.0, device=self.device)

        return loss

    def supervision_loss(self, n_points=200):
        """
        Supervision loss for separable potentials with known exact solutions.

        For HarmonicOscillator2D, the exact ground state momentum is:
        - p_x = -i*sqrt(m*omega_x)*x (depends only on x)
        - p_y = -i*sqrt(m*omega_y)*y (depends only on y)

        IMPORTANT: The momentum p = -i*alpha*x is ENERGY-INDEPENDENT for the
        harmonic oscillator ground state solution. We sample energies uniformly
        from the training range [E_min, E_max] to teach that p does not depend
        on E. This is critical for correct excited state energies.

        We supervise p_x along y=0 and p_y along x=0 to teach separability.

        Parameters
        ----------
        n_points : int
            Number of supervision points per direction

        Returns
        -------
        torch.Tensor
            Scalar loss value, or 0 if no exact solution known
        """
        if isinstance(self.potential, HarmonicOscillator2D):
            alpha_x = np.sqrt(self.potential.mass * self.potential.omega_x)
            alpha_y = np.sqrt(self.potential.mass * self.potential.omega_y)

            # Sample x points in complex plane (for supervising p_x at y=0)
            n_grid = int(np.sqrt(n_points))
            x_r = torch.linspace(-3.0, 3.0, n_grid, device=self.device)
            x_i = torch.linspace(-1.5, 1.5, n_grid, device=self.device)
            xr_grid, xi_grid = torch.meshgrid(x_r, x_i, indexing='ij')
            x_for_px = torch.complex(xr_grid.flatten(), xi_grid.flatten())
            y_zero = torch.zeros_like(x_for_px)  # y = 0

            # FIX: Sample energies uniformly from training range instead of fixed E_ground
            # This teaches that p = -i*alpha*x is independent of E
            n_px = len(x_for_px)
            E_random_px = torch.rand(n_px, device=self.device) * (self.E_max - self.E_min) + self.E_min
            E_for_px = torch.complex(E_random_px, torch.zeros_like(E_random_px))

            # p_x at y=0 should be -i*alpha_x*x (independent of E)
            p_x_exact = -1j * alpha_x * x_for_px
            p_x_pred, _ = self.net(x_for_px, y_zero, E_for_px)
            loss_px = torch.mean(torch.abs(p_x_pred - p_x_exact)**2)

            # Sample y points in complex plane (for supervising p_y at x=0)
            y_r = torch.linspace(-3.0, 3.0, n_grid, device=self.device)
            y_i = torch.linspace(-1.5, 1.5, n_grid, device=self.device)
            yr_grid, yi_grid = torch.meshgrid(y_r, y_i, indexing='ij')
            y_for_py = torch.complex(yr_grid.flatten(), yi_grid.flatten())
            x_zero = torch.zeros_like(y_for_py)  # x = 0

            # FIX: Sample energies uniformly from training range
            n_py = len(y_for_py)
            E_random_py = torch.rand(n_py, device=self.device) * (self.E_max - self.E_min) + self.E_min
            E_for_py = torch.complex(E_random_py, torch.zeros_like(E_random_py))

            # p_y at x=0 should be -i*alpha_y*y (independent of E)
            p_y_exact = -1j * alpha_y * y_for_py
            _, p_y_pred = self.net(x_zero, y_for_py, E_for_py)
            loss_py = torch.mean(torch.abs(p_y_pred - p_y_exact)**2)

            # Also supervise at some (x, y) != 0 to ensure full coverage
            # At (x, y) the momenta should still be p_x = -i*alpha_x*x, p_y = -i*alpha_y*y
            n_2d = n_grid // 2
            x_r2 = torch.linspace(-2.0, 2.0, n_2d, device=self.device)
            y_r2 = torch.linspace(-2.0, 2.0, n_2d, device=self.device)
            xr_2d, yr_2d = torch.meshgrid(x_r2, y_r2, indexing='ij')
            x_2d = torch.complex(xr_2d.flatten(), torch.zeros_like(xr_2d.flatten()))
            y_2d = torch.complex(yr_2d.flatten(), torch.zeros_like(yr_2d.flatten()))

            # FIX: Sample energies uniformly from training range
            n_2d_pts = len(x_2d)
            E_random_2d = torch.rand(n_2d_pts, device=self.device) * (self.E_max - self.E_min) + self.E_min
            E_2d = torch.complex(E_random_2d, torch.zeros_like(E_random_2d))

            p_x_exact_2d = -1j * alpha_x * x_2d
            p_y_exact_2d = -1j * alpha_y * y_2d
            p_x_pred_2d, p_y_pred_2d = self.net(x_2d, y_2d, E_2d)
            loss_2d = (torch.mean(torch.abs(p_x_pred_2d - p_x_exact_2d)**2) +
                       torch.mean(torch.abs(p_y_pred_2d - p_y_exact_2d)**2))

            return loss_px + loss_py + loss_2d

        elif isinstance(self.potential, CoupledHarmonicOscillator):
            # For coupled oscillator, supervision is more complex
            # Skip for now - rely on physics loss
            return torch.tensor(0.0, device=self.device)

        else:
            return torch.tensor(0.0, device=self.device)

    def sample_collocation_points(self, n_points, x_range=(-4, 4), y_range=(-4, 4),
                                   imag_range=(-0.5, 0.5)):
        """
        Sample collocation points in the 2D complex plane with energy.

        Parameters
        ----------
        n_points : int
            Number of points to sample
        x_range, y_range : tuple
            Range for real parts
        imag_range : tuple
            Range for imaginary parts

        Returns
        -------
        tuple of torch.Tensor
            (x_complex, y_complex, E_complex) collocation points
        """
        # Real parts
        x_real = torch.rand(n_points) * (x_range[1] - x_range[0]) + x_range[0]
        y_real = torch.rand(n_points) * (y_range[1] - y_range[0]) + y_range[0]

        # Imaginary parts (smaller for stability)
        x_imag = torch.rand(n_points) * (imag_range[1] - imag_range[0]) + imag_range[0]
        y_imag = torch.rand(n_points) * (imag_range[1] - imag_range[0]) + imag_range[0]

        # Energy: sample from training range (mostly real with small imaginary part)
        E_real = torch.rand(n_points) * (self.E_max - self.E_min) + self.E_min
        E_imag = torch.zeros(n_points)  # Energy is real

        x = torch.complex(x_real, x_imag).to(self.device)
        y = torch.complex(y_real, y_imag).to(self.device)
        E = torch.complex(E_real, E_imag).to(self.device)

        return x, y, E

    def train(self, n_epochs=20000, lr=1e-3, n_collocation=2000,
              lr_decay_step=5000, lr_decay_factor=0.5,
              physics_weight=1.0, curl_weight=0.5, asymptotic_weight=0.1,
              supervision_weight=1.0, verbose=True):
        """
        Train the PINN to learn the energy-conditioned quantum momentum.

        Parameters
        ----------
        n_epochs : int
            Number of training epochs
        lr : float
            Initial learning rate
        n_collocation : int
            Number of collocation points per batch
        lr_decay_step : int
            Decay learning rate after this many epochs
        lr_decay_factor : float
            Factor to multiply learning rate by
        physics_weight : float
            Weight for physics loss
        curl_weight : float
            Weight for curl loss
        asymptotic_weight : float
            Weight for asymptotic loss
        supervision_weight : float
            Weight for exact solution supervision (for separable potentials)
        verbose : bool
            Print training progress

        Returns
        -------
        float
            Final total loss
        """
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_decay_step, gamma=lr_decay_factor
        )

        self.loss_history = []
        self.physics_loss_history = []
        self.curl_loss_history = []
        self.asymptotic_loss_history = []
        self.supervision_loss_history = []

        # Estimate turning point for collocation range
        E_mid = (self.E_min + self.E_max) / 2
        tp_estimate = np.sqrt(2 * E_mid)

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Sample collocation points
            x_range = (-tp_estimate * 2.5, tp_estimate * 2.5)
            y_range = (-tp_estimate * 2.5, tp_estimate * 2.5)
            imag_range = (-tp_estimate * 0.3, tp_estimate * 0.3)

            x_coll, y_coll, E_coll = self.sample_collocation_points(
                n_collocation, x_range=x_range, y_range=y_range,
                imag_range=imag_range
            )

            # Compute losses
            phys_loss = self.physics_loss(x_coll, y_coll, E_coll)
            curl_loss_val = self.curl_loss(x_coll, y_coll, E_coll)
            asymp_loss = self.asymptotic_loss(x_coll, y_coll, E_coll)
            sup_loss = self.supervision_loss()

            # Total loss
            loss = (physics_weight * phys_loss +
                    curl_weight * curl_loss_val +
                    asymptotic_weight * asymp_loss +
                    supervision_weight * sup_loss)

            # Backpropagate
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            # Track losses
            self.loss_history.append(loss.item())
            self.physics_loss_history.append(phys_loss.item())
            self.curl_loss_history.append(curl_loss_val.item())
            self.asymptotic_loss_history.append(asymp_loss.item())
            self.supervision_loss_history.append(sup_loss.item())

            if verbose and (epoch % 2000 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch:5d}: total={loss.item():.2e}, "
                      f"phys={phys_loss.item():.2e}, curl={curl_loss_val.item():.2e}, "
                      f"sup={sup_loss.item():.2e}")

            # Early stopping
            if (phys_loss.item() < 1e-5 and curl_loss_val.item() < 1e-5 and
                sup_loss.item() < 1e-5):
                if verbose:
                    print(f"Converged at epoch {epoch}")
                break

        return self.loss_history[-1]

    def evaluate(self, x, y, E):
        """
        Evaluate the learned momentum (p_x, p_y) at positions (x, y) and energy E.

        Parameters
        ----------
        x, y, E : numpy.ndarray or torch.Tensor
            Complex coordinates and energy

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
            E_torch = torch.from_numpy(E.astype(np.complex64)).to(self.device)
        else:
            x_torch = x.to(self.device)
            y_torch = y.to(self.device)
            E_torch = E.to(self.device)

        with torch.no_grad():
            p_x, p_y = self.net(x_torch, y_torch, E_torch)

        if isinstance(x, np.ndarray):
            return p_x.cpu().numpy(), p_y.cpu().numpy()
        return p_x, p_y

    def verify_physics_residual(self, E, n_points=100):
        """
        Verify that the physics residual is small at a given energy.

        Parameters
        ----------
        E : float
            Energy to test
        n_points : int
            Number of test points

        Returns
        -------
        float
            Mean absolute physics residual
        """
        self.net.eval()

        # Sample on real axis only
        tp = np.sqrt(2 * E) if E > 0 else 2.0
        x = np.linspace(-tp * 2, tp * 2, n_points) + 0j
        y = np.linspace(-tp * 2, tp * 2, n_points) + 0j
        E_arr = np.full(n_points, E + 0j)

        p_x, p_y = self.evaluate(x, y, E_arr)

        # Compute residual manually
        h = 1e-5
        x_plus = x + h
        x_minus = x - h
        y_plus = y + h
        y_minus = y - h

        p_x_xplus, _ = self.evaluate(x_plus, y, E_arr)
        p_x_xminus, _ = self.evaluate(x_minus, y, E_arr)
        dp_x_dx = (p_x_xplus - p_x_xminus) / (2 * h)

        _, p_y_yplus = self.evaluate(x, y_plus, E_arr)
        _, p_y_yminus = self.evaluate(x, y_minus, E_arr)
        dp_y_dy = (p_y_yplus - p_y_yminus) / (2 * h)

        V = np.array([self.potential(xi.real, yi.real) for xi, yi in zip(x, y)])

        residual = (p_x**2 + p_y**2 +
                   1j * self.hbar * (dp_x_dx + dp_y_dy) -
                   2 * self.mass * (E - V))

        return np.mean(np.abs(residual))


class EBKQuantizer:
    """
    Find quantized energies via EBK (Einstein-Brillouin-Keller) conditions.

    The EBK quantization conditions require:
        J_x(E) = ℏ(n_x + 1/2)
        J_y(E) = ℏ(n_y + 1/2)

    where J_x and J_y are action integrals around closed loops in phase space.

    Parameters
    ----------
    pinn : QuantumHJPINN2DNonSep
        Trained PINN model
    potential : Potential2D
        The potential (for turning point calculations)
    hbar : float
        Reduced Planck constant (default: 1.0)
    mass : float
        Particle mass (default: 1.0)
    """

    def __init__(self, pinn, potential, hbar=1.0, mass=1.0):
        self.pinn = pinn
        self.potential = potential
        self.hbar = hbar
        self.mass = mass
        self.n_contour = 200  # Points for contour integration

    def _create_contour_x(self, E, y_fixed=0.0):
        """Create elliptical contour in complex x-plane at fixed y."""
        tp = self.potential.turning_points_x(E, y=y_fixed)
        if tp is None:
            # Use estimate based on energy
            x_est = np.sqrt(2 * E) if E > 0 else 1.0
            tp = (-x_est, x_est)

        x_left, x_right = tp
        center = (x_left + x_right) / 2
        a = 1.3 * (x_right - x_left) / 2
        b = 0.4 * a

        theta = np.linspace(0, 2 * np.pi, self.n_contour, endpoint=False)
        x_contour = center + a * np.cos(theta) + 1j * b * np.sin(theta)

        return x_contour

    def _create_contour_y(self, E, x_fixed=0.0):
        """Create elliptical contour in complex y-plane at fixed x."""
        tp = self.potential.turning_points_y(E, x=x_fixed)
        if tp is None:
            y_est = np.sqrt(2 * E) if E > 0 else 1.0
            tp = (-y_est, y_est)

        y_left, y_right = tp
        center = (y_left + y_right) / 2
        a = 1.3 * (y_right - y_left) / 2
        b = 0.4 * a

        theta = np.linspace(0, 2 * np.pi, self.n_contour, endpoint=False)
        y_contour = center + a * np.cos(theta) + 1j * b * np.sin(theta)

        return y_contour

    def compute_action_x(self, E, y_fixed=0.0, method='analytic'):
        """
        Compute x-direction action integral J_x.

        Parameters
        ----------
        E : float
            Energy
        y_fixed : float
            Fixed y-coordinate (default: 0)
        method : str
            'analytic' extracts coefficient from learned p (default for HO)
            'contour' uses numerical contour integration

        Returns
        -------
        float
            Action integral J_x / ℏ
        """
        if method == 'analytic' and isinstance(self.potential, (HarmonicOscillator2D, CoupledHarmonicOscillator)):
            return self._compute_action_x_analytic(E, y_fixed)
        else:
            return self._compute_action_x_contour(E, y_fixed)

    def _compute_action_x_analytic(self, E, y_fixed=0.0):
        """
        Compute J_x analytically from the learned momentum coefficient.

        For harmonic oscillator with p_x = -i * alpha * x, we have:
            J_x / ℏ = alpha² / (2 * m * omega)

        For exact alpha = sqrt(m*omega), this gives J_x/ℏ = 0.5 for ground state.
        """
        # Evaluate at reference point to extract alpha
        x_ref = np.array([1.0 + 0j])
        y_ref = np.array([y_fixed + 0j])
        E_ref = np.array([E + 0j])
        p_x, _ = self.pinn.evaluate(x_ref, y_ref, E_ref)
        p_x = p_x[0]

        # Extract alpha from p_x = -i * alpha * x at x=1
        # p_x = -i * alpha => alpha = i * p_x
        alpha = 1j * p_x

        # For harmonic oscillator: J_x / ℏ = |alpha|² / (2 * m * omega)
        if isinstance(self.potential, HarmonicOscillator2D):
            omega = self.potential.omega_x
            mass = self.potential.mass
        elif isinstance(self.potential, CoupledHarmonicOscillator):
            # For coupled oscillator, use effective frequency
            omega = 1.0  # Base frequency
            mass = self.potential.mass
        else:
            omega = 1.0
            mass = 1.0

        alpha_mag_sq = abs(alpha) ** 2
        J_over_hbar = alpha_mag_sq / (2 * mass * omega)

        return J_over_hbar

    def _compute_action_x_contour(self, E, y_fixed=0.0):
        """Compute J_x using numerical contour integration."""
        x_contour = self._create_contour_x(E, y_fixed)
        y_arr = np.full_like(x_contour, y_fixed + 0j)
        E_arr = np.full_like(x_contour, E + 0j)

        p_x, _ = self.pinn.evaluate(x_contour, y_arr, E_arr)

        # Trapezoidal integration
        integral = 0j
        n = len(x_contour)
        for i in range(n):
            dx = x_contour[(i + 1) % n] - x_contour[i]
            p_avg = (p_x[i] + p_x[(i + 1) % n]) / 2
            integral += p_avg * dx

        J_x = integral / (2 * np.pi * 1j)

        return abs(J_x.imag) / self.hbar

    def compute_action_y(self, E, x_fixed=0.0, method='analytic'):
        """
        Compute y-direction action integral J_y.

        Parameters
        ----------
        E : float
            Energy
        x_fixed : float
            Fixed x-coordinate (default: 0)
        method : str
            'analytic' extracts coefficient from learned p (default for HO)
            'contour' uses numerical contour integration

        Returns
        -------
        float
            Action integral J_y / ℏ
        """
        if method == 'analytic' and isinstance(self.potential, (HarmonicOscillator2D, CoupledHarmonicOscillator)):
            return self._compute_action_y_analytic(E, x_fixed)
        else:
            return self._compute_action_y_contour(E, x_fixed)

    def _compute_action_y_analytic(self, E, x_fixed=0.0):
        """
        Compute J_y analytically from the learned momentum coefficient.
        """
        # Evaluate at reference point to extract alpha
        x_ref = np.array([x_fixed + 0j])
        y_ref = np.array([1.0 + 0j])
        E_ref = np.array([E + 0j])
        _, p_y = self.pinn.evaluate(x_ref, y_ref, E_ref)
        p_y = p_y[0]

        # Extract alpha from p_y = -i * alpha * y at y=1
        alpha = 1j * p_y

        # For harmonic oscillator: J_y / ℏ = |alpha|² / (2 * m * omega)
        if isinstance(self.potential, HarmonicOscillator2D):
            omega = self.potential.omega_y
            mass = self.potential.mass
        elif isinstance(self.potential, CoupledHarmonicOscillator):
            omega = 1.0
            mass = self.potential.mass
        else:
            omega = 1.0
            mass = 1.0

        alpha_mag_sq = abs(alpha) ** 2
        J_over_hbar = alpha_mag_sq / (2 * mass * omega)

        return J_over_hbar

    def _compute_action_y_contour(self, E, x_fixed=0.0):
        """Compute J_y using numerical contour integration."""
        y_contour = self._create_contour_y(E, x_fixed)
        x_arr = np.full_like(y_contour, x_fixed + 0j)
        E_arr = np.full_like(y_contour, E + 0j)

        _, p_y = self.pinn.evaluate(x_arr, y_contour, E_arr)

        # Trapezoidal integration
        integral = 0j
        n = len(y_contour)
        for i in range(n):
            dy = y_contour[(i + 1) % n] - y_contour[i]
            p_avg = (p_y[i] + p_y[(i + 1) % n]) / 2
            integral += p_avg * dy

        J_y = integral / (2 * np.pi * 1j)

        return abs(J_y.imag) / self.hbar

    def quantization_residual(self, E, n_x, n_y, x_fixed=0.0, y_fixed=0.0):
        """
        Compute quantization residual for state (n_x, n_y) at energy E.

        Returns sum of squared deviations from quantization conditions:
            (J_x/ℏ - (n_x + 1/2))² + (J_y/ℏ - (n_y + 1/2))²

        Parameters
        ----------
        E : float
            Energy
        n_x, n_y : int
            Target quantum numbers
        x_fixed, y_fixed : float
            Fixed coordinates for action integrals

        Returns
        -------
        float
            Quantization residual
        """
        J_x = self.compute_action_x(E, y_fixed=y_fixed)
        J_y = self.compute_action_y(E, x_fixed=x_fixed)

        target_x = n_x + 0.5
        target_y = n_y + 0.5

        residual = (J_x - target_x)**2 + (J_y - target_y)**2

        return residual

    def find_quantized_energy(self, n_x, n_y, E_min, E_max, tol=1e-3):
        """
        Find the quantized energy for state (n_x, n_y).

        Uses minimization to find E where both quantization conditions
        are satisfied.

        Parameters
        ----------
        n_x, n_y : int
            Target quantum numbers
        E_min, E_max : float
            Energy search range
        tol : float
            Tolerance for residual

        Returns
        -------
        dict
            Result dictionary with 'E', 'J_x_over_hbar', 'J_y_over_hbar',
            'residual', 'success'
        """
        def objective(E):
            return self.quantization_residual(E, n_x, n_y)

        # Grid search for initial guess
        E_grid = np.linspace(E_min, E_max, 20)
        residuals = [objective(E) for E in E_grid]
        E_init = E_grid[np.argmin(residuals)]

        # Refine with minimize_scalar
        result = minimize_scalar(
            objective,
            bounds=(E_min, E_max),
            method='bounded'
        )

        E_opt = result.x
        J_x = self.compute_action_x(E_opt)
        J_y = self.compute_action_y(E_opt)
        residual = result.fun

        return {
            'n_x': n_x,
            'n_y': n_y,
            'E': E_opt,
            'J_x_over_hbar': J_x,
            'J_y_over_hbar': J_y,
            'residual': residual,
            'success': residual < tol
        }

    def find_quantized_energies(self, E_min, E_max, n_max=3):
        """
        Find all quantized energies in range for n_x, n_y ≤ n_max.

        Parameters
        ----------
        E_min, E_max : float
            Energy search range
        n_max : int
            Maximum quantum number to search

        Returns
        -------
        list of dict
            List of results for each found state
        """
        results = []
        for n_x in range(n_max + 1):
            for n_y in range(n_max + 1):
                # Estimate energy for this state
                E_est = (n_x + n_y + 1) * self.hbar  # Rough harmonic estimate
                search_min = max(E_min, E_est * 0.5)
                search_max = min(E_max, E_est * 2.0)

                if search_min >= search_max:
                    continue

                result = self.find_quantized_energy(n_x, n_y, search_min, search_max)
                if result['success']:
                    results.append(result)

        # Sort by energy
        results.sort(key=lambda r: r['E'])

        return results


class NonSepPINNSolver:
    """
    High-level solver for non-separable 2D quantum systems.

    Combines PINN training with EBK quantization and DVR validation.

    Parameters
    ----------
    potential : Potential2D
        Non-separable 2D potential
    E_range : tuple
        (E_min, E_max) energy range
    hbar : float
        Reduced Planck constant (default: 1.0)
    mass : float
        Particle mass (default: 1.0)
    device : str
        PyTorch device (default: auto-detect)
    """

    def __init__(self, potential, E_range, hbar=1.0, mass=1.0, device=None):
        self.potential = potential
        self.E_range = E_range
        self.hbar = hbar
        self.mass = mass

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.pinn = None
        self.quantizer = None

    def train(self, n_epochs=20000, verbose=True, **kwargs):
        """
        Train the PINN solver.

        Parameters
        ----------
        n_epochs : int
            Number of training epochs
        verbose : bool
            Print progress
        **kwargs
            Additional arguments for QuantumHJPINN2DNonSep.train()

        Returns
        -------
        QuantumHJPINN2DNonSep
            Trained PINN
        """
        self.pinn = QuantumHJPINN2DNonSep(
            potential=self.potential,
            E_range=self.E_range,
            hbar=self.hbar,
            mass=self.mass,
            device=self.device
        )

        self.pinn.train(n_epochs=n_epochs, verbose=verbose, **kwargs)

        self.quantizer = EBKQuantizer(
            self.pinn, self.potential, self.hbar, self.mass
        )

        return self.pinn

    def find_energies(self, n_max=3):
        """
        Find quantized energies up to n_x, n_y = n_max.

        Parameters
        ----------
        n_max : int
            Maximum quantum number

        Returns
        -------
        list of dict
            Results for each found state
        """
        if self.quantizer is None:
            raise ValueError("Must call train() first")

        return self.quantizer.find_quantized_energies(
            self.E_range[0], self.E_range[1], n_max=n_max
        )

    def compare_with_dvr(self, dvr_energies, n_states=10):
        """
        Compare PINN energies with DVR reference.

        Parameters
        ----------
        dvr_energies : array-like
            DVR reference eigenvalues
        n_states : int
            Number of states to compare

        Returns
        -------
        dict
            Comparison results with errors
        """
        pinn_results = self.find_energies(n_max=int(np.sqrt(n_states)) + 1)

        # Match states by energy ordering
        pinn_energies = sorted([r['E'] for r in pinn_results])[:n_states]
        dvr_energies = np.array(dvr_energies)[:n_states]

        errors = np.abs(np.array(pinn_energies) - dvr_energies)
        rel_errors = errors / dvr_energies

        return {
            'pinn_energies': pinn_energies,
            'dvr_energies': list(dvr_energies),
            'absolute_errors': list(errors),
            'relative_errors': list(rel_errors),
            'max_error': np.max(errors),
            'mean_error': np.mean(errors),
            'max_rel_error': np.max(rel_errors),
            'mean_rel_error': np.mean(rel_errors)
        }
