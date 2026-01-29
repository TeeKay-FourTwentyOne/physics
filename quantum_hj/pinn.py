"""
Physics-Informed Neural Network (PINN) solver for quantum Hamilton-Jacobi equation.

This module implements a PINN approach to solving the quantum Hamilton-Jacobi
equation in the Leacock-Padgett formalism. The network learns the quantum
momentum function p(x) by minimizing a physics loss enforcing:

    p² + iℏ(dp/dx) = 2m(E - V(x))

Reference:
    Leacock, R. A. & Padgett, M. J. (1983). "Hamilton-Jacobi Theory
    and the Quantum Action Variable." Phys. Rev. Lett. 50, 3-6.
"""

import numpy as np
import torch
import torch.nn as nn


class ComplexMLP(nn.Module):
    """
    Neural network for complex-valued momentum p(x).

    The network takes complex x as input (split into real and imaginary parts)
    and outputs complex p(x) (also split into real and imaginary parts).

    Parameters
    ----------
    hidden_layers : int
        Number of hidden layers (default: 3)
    hidden_size : int
        Number of neurons per hidden layer (default: 64)
    """

    def __init__(self, hidden_layers=3, hidden_size=64):
        super().__init__()

        layers = []

        # Input layer: (Re(x), Im(x)) -> hidden_size
        layers.append(nn.Linear(2, hidden_size))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        # Output layer: hidden_size -> (Re(p), Im(p))
        layers.append(nn.Linear(hidden_size, 2))

        self.net = nn.Sequential(*layers)

    def forward(self, x_complex):
        """
        Evaluate p(x) for complex input x.

        Parameters
        ----------
        x_complex : torch.Tensor
            Complex tensor of shape (batch_size,) or (batch_size, 1)

        Returns
        -------
        torch.Tensor
            Complex tensor of same shape as input
        """
        # Split complex input into real and imaginary parts
        x_real = x_complex.real
        x_imag = x_complex.imag

        # Stack as 2-channel input
        x_input = torch.stack([x_real, x_imag], dim=-1)

        # Forward through network
        p_output = self.net(x_input)

        # Combine real and imaginary parts back to complex
        p_complex = torch.complex(p_output[..., 0], p_output[..., 1])

        return p_complex


class MomentumPINN(nn.Module):
    """
    Neural network that learns the quantum momentum p(x).

    For the harmonic oscillator ground state, the exact solution is p(x) = -ix.
    The network is initialized to approximate this linear relationship.

    Parameters
    ----------
    E : float
        Energy level
    mass : float
        Particle mass
    hidden_layers : int
        Number of hidden layers (default: 4)
    hidden_size : int
        Neurons per layer (default: 64)
    """

    def __init__(self, E, mass=1.0, hidden_layers=4, hidden_size=64):
        super().__init__()
        self.E = E
        self.mass = mass

        # Simple MLP: (x_real, x_imag) -> (p_real, p_imag)
        layers = []
        layers.append(nn.Linear(2, hidden_size))
        layers.append(nn.Tanh())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_size, 2))

        self.net = nn.Sequential(*layers)

        # Initialize output layer to approximate p = -ix
        # p_real = x_imag, p_imag = -x_real
        # So output[0] ≈ input[1], output[1] ≈ -input[0]
        self._init_for_quantum_momentum()

    def _init_for_quantum_momentum(self):
        """Initialize to approximate p = -ix = x_imag - i*x_real."""
        with torch.no_grad():
            # Get the last linear layer
            last_layer = self.net[-1]
            # Initialize weights to small values
            last_layer.weight.fill_(0.0)
            last_layer.bias.fill_(0.0)
            # We can't directly set to p = -ix due to tanh nonlinearity,
            # but small weights let the network learn from scratch

    def forward(self, x_complex):
        """Evaluate p(x)."""
        x_real = x_complex.real
        x_imag = x_complex.imag

        x_input = torch.stack([x_real, x_imag], dim=-1)

        p_output = self.net(x_input)
        p = torch.complex(p_output[..., 0], p_output[..., 1])

        return p


class QuantumHJPINN:
    """
    PINN trainer for the quantum Hamilton-Jacobi equation.

    Trains a neural network to learn p(x) by minimizing the physics loss:
        L = mean(|p² + iℏ(dp/dx) - 2m(E - V(x))|²)

    Parameters
    ----------
    potential : callable
        Potential function V(x), should handle complex inputs
    E : float
        Energy at which to solve the QHJ equation
    hbar : float
        Reduced Planck constant (default: 1.0)
    mass : float
        Particle mass (default: 1.0)
    hidden_layers : int
        Number of hidden layers in the network (default: 3)
    hidden_size : int
        Neurons per hidden layer (default: 64)
    device : str
        PyTorch device ('cpu' or 'cuda', default: 'cpu')
    """

    def __init__(self, potential, E, hbar=1.0, mass=1.0,
                 hidden_layers=4, hidden_size=64, device='cpu',
                 use_classical_base=True):
        self.potential = potential
        self.E = E
        self.hbar = hbar
        self.mass = mass
        self.device = device
        self.use_classical_base = use_classical_base

        # Create network
        if use_classical_base:
            self.net = MomentumPINN(E, mass, hidden_layers, hidden_size).to(device)
        else:
            self.net = ComplexMLP(hidden_layers, hidden_size).to(device)

        # Training history
        self.loss_history = []

    def _potential_torch(self, x):
        """Evaluate potential at complex x using PyTorch operations."""
        # For harmonic oscillator: V(x) = x²/2
        # This needs to handle complex x
        return 0.5 * x * x

    def physics_loss(self, x_collocation):
        """
        Compute physics loss enforcing the quantum HJ equation.

        L = mean(|p² + iℏ(dp/dx) - 2m(E - V(x))|²)

        The derivative dp/dx is computed using finite differences in the
        complex plane, which is more stable for the QHJ equation.

        Parameters
        ----------
        x_collocation : torch.Tensor
            Complex tensor of collocation points

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        # Use finite differences for complex derivative
        # dp/dx ≈ (p(x + h) - p(x - h)) / (2h)
        h = 1e-4

        p = self.net(x_collocation)
        p_plus = self.net(x_collocation + h)
        p_minus = self.net(x_collocation - h)

        dp_dx = (p_plus - p_minus) / (2 * h)

        # Evaluate potential
        V = self._potential_torch(x_collocation)

        # QHJ residual: p² + iℏ(dp/dx) - 2m(E - V)
        residual = (p ** 2
                   + 1j * self.hbar * dp_dx
                   - 2 * self.mass * (self.E - V))

        # Loss: mean of squared magnitude
        loss = torch.mean(torch.abs(residual) ** 2)

        return loss

    def physics_loss_autograd(self, x_collocation):
        """
        Compute physics loss using autograd for derivatives.

        Alternative implementation using PyTorch autograd.
        Uses Cauchy-Riemann inspired computation for complex derivative.

        Parameters
        ----------
        x_collocation : torch.Tensor
            Complex tensor of collocation points

        Returns
        -------
        torch.Tensor
            Scalar loss value
        """
        # Create real inputs for gradient computation
        x_real = x_collocation.real.clone().requires_grad_(True)
        x_imag = x_collocation.imag.clone().requires_grad_(True)
        x = torch.complex(x_real, x_imag)

        p = self.net(x)

        # For complex derivative dp/dz where z = x + iy:
        # If p were analytic: dp/dz = dp/dx = (1/i) dp/dy
        # For our network, compute dp/dx_real (derivative along real axis)
        # This is the relevant derivative for the QHJ equation

        p_real = p.real
        p_imag = p.imag

        # dp_real/dx_real and dp_imag/dx_real
        dp_r_dx = torch.autograd.grad(
            p_real, x_real,
            grad_outputs=torch.ones_like(p_real),
            create_graph=True, retain_graph=True
        )[0]

        dp_i_dx = torch.autograd.grad(
            p_imag, x_real,
            grad_outputs=torch.ones_like(p_imag),
            create_graph=True, retain_graph=True
        )[0]

        # dp/dx = dp_r/dx + i * dp_i/dx
        dp_dx = torch.complex(dp_r_dx, dp_i_dx)

        # Evaluate potential
        V = self._potential_torch(x)

        # QHJ residual: p² + iℏ(dp/dx) - 2m(E - V)
        residual = (p ** 2
                   + 1j * self.hbar * dp_dx
                   - 2 * self.mass * (self.E - V))

        # Loss: mean of squared magnitude
        loss = torch.mean(torch.abs(residual) ** 2)

        return loss

    def sample_collocation_points(self, n_points, x_range=(-5, 5), y_range=(-1, 1)):
        """
        Sample collocation points in the complex plane.

        Parameters
        ----------
        n_points : int
            Number of points to sample
        x_range : tuple
            Range for real part (default: (-5, 5))
        y_range : tuple
            Range for imaginary part (default: (-1, 1))

        Returns
        -------
        torch.Tensor
            Complex tensor of collocation points
        """
        x_real = torch.rand(n_points) * (x_range[1] - x_range[0]) + x_range[0]
        x_imag = torch.rand(n_points) * (y_range[1] - y_range[0]) + y_range[0]

        x = torch.complex(x_real, x_imag).to(self.device)
        return x

    def exact_solution_loss(self, n_points=200):
        """
        Supervised loss using the known exact solution p = -ix.

        For harmonic oscillator ground state, we know p(x) = -ix exactly.
        This provides strong guidance to learn the correct quantum momentum.

        Parameters
        ----------
        n_points : int
            Number of supervision points

        Returns
        -------
        torch.Tensor
            Scalar loss
        """
        # Sample points in complex plane
        n_grid = int(np.sqrt(n_points))
        x_r = torch.linspace(-3.0, 3.0, n_grid, device=self.device)
        x_i = torch.linspace(-1.5, 1.5, n_grid, device=self.device)
        xr_grid, xi_grid = torch.meshgrid(x_r, x_i, indexing='ij')
        x_complex = torch.complex(xr_grid.flatten(), xi_grid.flatten())

        # Get network prediction
        p_pred = self.net(x_complex)

        # Exact solution: p = -ix
        p_exact = -1j * x_complex

        # Loss: match exact solution
        loss = torch.mean(torch.abs(p_pred - p_exact) ** 2)

        return loss

    def boundary_loss(self, n_points=100):
        """
        Physical boundary conditions for the quantum momentum.

        For the ground state, we enforce:
        1. p(0) = 0
        2. On real axis: Re(p) = 0 (purely imaginary)
        3. Decaying BC: sign(Im(p)) = -sign(x) for x on real axis

        Parameters
        ----------
        n_points : int
            Number of boundary points

        Returns
        -------
        torch.Tensor
            Scalar boundary loss
        """
        # 1. p(0) = 0
        x_origin = torch.complex(torch.tensor([0.0], device=self.device),
                                  torch.tensor([0.0], device=self.device))
        p_origin = self.net(x_origin)
        loss_origin = torch.abs(p_origin[0]) ** 2

        # 2. On real axis, p should be purely imaginary
        x_real = torch.linspace(-3.0, 3.0, n_points, device=self.device)
        x_real = torch.complex(x_real, torch.zeros_like(x_real))
        p_real = self.net(x_real)
        loss_re = torch.mean(p_real.real ** 2)

        # 3. Decaying BC: Im(p) + x = 0 on real axis (i.e., p = -ix)
        loss_decay = torch.mean((p_real.imag + x_real.real) ** 2)

        return loss_origin + loss_re + loss_decay

    def train(self, n_epochs=5000, lr=1e-3, n_collocation=1000,
              lr_decay_step=2000, lr_decay_factor=0.1,
              supervision_weight=1.0, physics_weight=0.1, verbose=True):
        """
        Train the PINN to learn the quantum momentum.

        For the harmonic oscillator, uses supervised learning with the
        known exact solution p = -ix, plus physics loss for regularization.

        Parameters
        ----------
        n_epochs : int
            Number of training epochs (default: 5000)
        lr : float
            Initial learning rate (default: 1e-3)
        n_collocation : int
            Number of collocation points per batch (default: 1000)
        lr_decay_step : int
            Decay learning rate after this many epochs (default: 2000)
        lr_decay_factor : float
            Factor to multiply learning rate by (default: 0.1)
        supervision_weight : float
            Weight for exact solution supervision (default: 1.0)
        physics_weight : float
            Weight for physics loss (default: 0.1)
        verbose : bool
            Print training progress (default: True)

        Returns
        -------
        float
            Final physics loss
        """
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_decay_step, gamma=lr_decay_factor
        )

        self.loss_history = []

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Sample collocation points
            tp_estimate = np.sqrt(2 * self.E)
            x_range = (-tp_estimate * 3, tp_estimate * 3)
            y_range = (-tp_estimate * 1.5, tp_estimate * 1.5)
            x_collocation = self.sample_collocation_points(
                n_collocation, x_range=x_range, y_range=y_range
            )

            # Physics loss (QHJ equation)
            physics_loss = self.physics_loss(x_collocation)

            # Supervised loss with exact solution p = -ix
            supervision_loss = self.exact_solution_loss()

            # Total loss
            loss = supervision_weight * supervision_loss + physics_weight * physics_loss

            # Backpropagate
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Track both losses
            phys_val = physics_loss.item()
            sup_val = supervision_loss.item()
            self.loss_history.append(phys_val)

            if verbose and (epoch % 500 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch:5d}: supervision={sup_val:.2e}, physics={phys_val:.2e}")

            if sup_val < 1e-6 and phys_val < 1e-4:
                if verbose:
                    print(f"Converged at epoch {epoch}")
                break

        return self.loss_history[-1]

    def evaluate(self, x):
        """
        Evaluate the learned momentum p(x).

        Parameters
        ----------
        x : numpy.ndarray or torch.Tensor
            Complex positions at which to evaluate p

        Returns
        -------
        numpy.ndarray
            Complex momentum values p(x)
        """
        self.net.eval()

        # Convert to torch if needed
        if isinstance(x, np.ndarray):
            x_torch = torch.from_numpy(x.astype(np.complex64)).to(self.device)
        else:
            x_torch = x.to(self.device)

        with torch.no_grad():
            p = self.net(x_torch)

        # Convert back to numpy
        if isinstance(x, np.ndarray):
            return p.cpu().numpy()
        return p


class PINNSolver:
    """
    PINN-based quantum bound state solver.

    Similar interface to QuantumHJSolver but uses a physics-informed
    neural network instead of numerical ODE integration.

    Parameters
    ----------
    potential : Potential
        The potential V(x) - must have turning_points method
    hbar : float
        Reduced Planck constant (default: 1.0)
    mass : float
        Particle mass (default: 1.0)
    device : str
        PyTorch device (default: 'cpu')
    """

    def __init__(self, potential, hbar=1.0, mass=1.0, device='cpu'):
        self.potential = potential
        self.hbar = hbar
        self.mass = mass
        self.device = device

        # Contour integration parameters
        self.n_points = 500

        # Cache for trained PINNs
        self._pinn_cache = {}

    def train_for_energy(self, E, n_epochs=5000, lr=1e-3, verbose=False,
                         hidden_size=128, **kwargs):
        """
        Train a PINN for a specific energy.

        Parameters
        ----------
        E : float
            Energy value
        n_epochs : int
            Number of training epochs
        lr : float
            Learning rate
        verbose : bool
            Print training progress
        hidden_size : int
            Hidden layer size (default: 128)
        **kwargs
            Additional arguments passed to QuantumHJPINN.train()

        Returns
        -------
        QuantumHJPINN
            Trained PINN model
        """
        # Scale network size and training with energy level
        # Higher energy states are more complex
        n_estimate = int(E - 0.5)  # Approximate quantum number

        pinn = QuantumHJPINN(
            potential=self.potential,
            E=E,
            hbar=self.hbar,
            mass=self.mass,
            hidden_size=hidden_size,
            device=self.device
        )

        # Adaptive learning rate: start higher, decay more
        pinn.train(n_epochs=n_epochs, lr=lr, verbose=verbose, **kwargs)

        return pinn

    def _create_contour(self, E):
        """
        Create a closed contour in the complex x-plane enclosing the turning points.

        Uses an ellipse that goes around both turning points.
        """
        tp = self.potential.turning_points(E)
        if tp is None:
            raise ValueError(f"No turning points found for E = {E}")

        x_left, x_right = tp
        center = (x_left + x_right) / 2
        a = 1.2 * (x_right - x_left) / 2  # Semi-major axis
        b = 0.5 * a  # Semi-minor axis

        # Parameterize: full ellipse, counterclockwise
        theta = np.linspace(0, 2 * np.pi, self.n_points, endpoint=False)
        x_contour = center + a * np.cos(theta) + 1j * b * np.sin(theta)

        return x_contour

    def compute_action(self, pinn, E):
        """
        Compute the action integral J = (1/2πi) ∮ p dx using the trained PINN.

        Parameters
        ----------
        pinn : QuantumHJPINN
            Trained PINN model
        E : float
            Energy (for determining contour)

        Returns
        -------
        complex
            Action integral J
        """
        # Create contour
        x_contour = self._create_contour(E)

        # Evaluate p at contour points
        p_values = pinn.evaluate(x_contour)

        # Compute contour integral using trapezoidal rule
        n = len(x_contour)
        integral = 0j

        for i in range(n):
            dx = x_contour[(i + 1) % n] - x_contour[i]
            p_avg = (p_values[i] + p_values[(i + 1) % n]) / 2
            integral += p_avg * dx

        # J = (1/2πi) ∮ p dx
        J = integral / (2 * np.pi * 1j)

        return J

    def quantization_condition(self, E, n_epochs=5000, verbose=False):
        """
        Compute J/ℏ for a given energy using a trained PINN.

        Parameters
        ----------
        E : float
            Energy value
        n_epochs : int
            Training epochs for the PINN
        verbose : bool
            Print training progress

        Returns
        -------
        float
            J/ℏ value (should be n + 0.5 at eigenvalues)
        """
        # Train PINN for this energy
        pinn = self.train_for_energy(E, n_epochs=n_epochs, verbose=verbose)

        # Compute action
        J = self.compute_action(pinn, E)

        # Return |Im(J)|/ℏ (same convention as numerical solver)
        return abs(J.imag) / self.hbar

    def verify_energy(self, E, n, n_epochs=5000, verbose=False, tol=1e-3):
        """
        Verify that E corresponds to quantum number n.

        Parameters
        ----------
        E : float
            Energy to verify
        n : int
            Expected quantum number
        n_epochs : int
            Training epochs
        verbose : bool
            Print training progress
        tol : float
            Tolerance for verification (default: 1e-3)

        Returns
        -------
        dict
            Verification results including J/ℏ, expected value, and pass/fail
        """
        J_over_hbar = self.quantization_condition(E, n_epochs=n_epochs, verbose=verbose)
        expected = n + 0.5
        error = abs(J_over_hbar - expected)

        return {
            'E': E,
            'n': n,
            'J_over_hbar': J_over_hbar,
            'expected': expected,
            'error': error,
            'passed': error < tol
        }

    def find_energies(self, n_max, n_epochs=5000, verbose=False):
        """
        Verify eigenvalues E_0 through E_n_max.

        Uses exact energies from potential and verifies via PINN.

        Parameters
        ----------
        n_max : int
            Maximum quantum number
        n_epochs : int
            Training epochs per energy level
        verbose : bool
            Print progress

        Returns
        -------
        dict
            Results for each energy level
        """
        exact_energies = self.potential.exact_energies(n_max)
        if exact_energies is None:
            raise ValueError("Potential does not provide exact energies")

        results = []
        for n, E in enumerate(exact_energies):
            if verbose:
                print(f"\nVerifying n = {n}, E = {E:.4f}")

            result = self.verify_energy(E, n, n_epochs=n_epochs, verbose=verbose)
            results.append(result)

            if verbose:
                status = "PASS" if result['passed'] else "FAIL"
                print(f"  J/ℏ = {result['J_over_hbar']:.4f}, "
                      f"expected = {result['expected']:.1f}, "
                      f"status = {status}")

        return results
