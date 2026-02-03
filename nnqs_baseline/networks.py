"""
Neural network architectures for NNQS.

The log-amplitude representation psi(x) = exp(f(x)) ensures:
- Positive wavefunction (valid for bosonic ground states)
- Numerical stability
- Automatic normalizability
"""

import torch
import torch.nn as nn


class LogAmplitudeNetwork(nn.Module):
    """
    Neural network that outputs log|psi(x)| = f(x).

    The wavefunction is psi(x) = exp(f(x)), so:
    - grad(psi)/psi = grad(f)
    - laplacian(psi)/psi = laplacian(f) + |grad(f)|^2

    Args:
        input_dim: Number of spatial dimensions
        hidden_dims: List of hidden layer sizes
        activation: Activation function (default: tanh)
    """

    def __init__(self, input_dim=1, hidden_dims=None, activation=None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]

        if activation is None:
            activation = nn.Tanh()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Build network
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation)
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # Output: scalar log|psi|

        self.net = nn.Sequential(*layers)

        # Initialize final layer to near-zero so initial psi ~ 1 (uniform)
        # This prevents initial |psi|^2 from being too peaked or too flat
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        """
        Compute log|psi(x)|.

        Args:
            x: Positions, shape (batch,) or (batch, dim)

        Returns:
            f: log|psi|, shape (batch,)
        """
        # Ensure x has shape (batch, dim)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        return self.net(x).squeeze(-1)

    def log_prob(self, x):
        """
        Compute log|psi(x)|^2 = 2*f(x) for sampling.

        Args:
            x: Positions, shape (batch,) or (batch, dim)

        Returns:
            log_prob: 2*log|psi|, shape (batch,)
        """
        return 2.0 * self.forward(x)

    def get_momentum(self, x):
        """
        Extract momentum field from wavefunction for comparison with LP+PINN.

        p = -i*hbar * (grad psi)/psi = -i*hbar * grad(f)

        For real wavefunction (ground state), p is pure imaginary:
        p_real = 0, p_imag = -grad(f)

        Args:
            x: Positions, shape (batch, dim), requires_grad should be True

        Returns:
            p_real: Real part of momentum, shape (batch, dim)
            p_imag: Imaginary part of momentum, shape (batch, dim)
        """
        if not x.requires_grad:
            x = x.requires_grad_(True)

        f = self.forward(x)
        grad_f = torch.autograd.grad(
            f.sum(), x, create_graph=True, retain_graph=True
        )[0]

        # p = -i * grad_f (for hbar=1)
        # p_real = 0, p_imag = -grad_f
        p_real = torch.zeros_like(grad_f)
        p_imag = -grad_f

        return p_real, p_imag


class ComplexLogAmplitudeNetwork(nn.Module):
    """
    Network for complex wavefunction: psi = exp(f + i*g).

    Needed for excited states or systems with complex-valued ground states.
    Outputs both log-amplitude f and phase g.

    Args:
        input_dim: Number of spatial dimensions
        hidden_dims: List of hidden layer sizes
    """

    def __init__(self, input_dim=1, hidden_dims=None):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        # Shared backbone
        backbone_layers = []
        prev_dim = input_dim
        for h in hidden_dims[:-1]:  # All but last hidden layer
            backbone_layers.append(nn.Linear(prev_dim, h))
            backbone_layers.append(nn.Tanh())
            prev_dim = h

        self.backbone = nn.Sequential(*backbone_layers)

        # Separate heads for amplitude and phase
        last_hidden = hidden_dims[-1] if hidden_dims else input_dim
        self.amplitude_head = nn.Sequential(
            nn.Linear(prev_dim, last_hidden),
            nn.Tanh(),
            nn.Linear(last_hidden, 1)
        )
        self.phase_head = nn.Sequential(
            nn.Linear(prev_dim, last_hidden),
            nn.Tanh(),
            nn.Linear(last_hidden, 1)
        )

        # Initialize to near-zero
        nn.init.zeros_(self.amplitude_head[-1].weight)
        nn.init.zeros_(self.amplitude_head[-1].bias)
        nn.init.zeros_(self.phase_head[-1].weight)
        nn.init.zeros_(self.phase_head[-1].bias)

    def forward(self, x):
        """
        Compute (log|psi|, phase).

        Args:
            x: Positions, shape (batch,) or (batch, dim)

        Returns:
            f: log|psi|, shape (batch,)
            g: phase angle, shape (batch,)
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        features = self.backbone(x)
        f = self.amplitude_head(features).squeeze(-1)
        g = self.phase_head(features).squeeze(-1)

        return f, g
