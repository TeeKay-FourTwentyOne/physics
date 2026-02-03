"""
Local energy computation for variational Monte Carlo.

The local energy is E_loc(x) = (H*psi)(x) / psi(x), where H = -1/2 * nabla^2 + V.

For psi = exp(f), this simplifies to:
    E_loc = -1/2 * (laplacian(f) + |grad(f)|^2) + V(x)

The variational energy is E = <E_loc>_{|psi|^2}, which is estimated
by sampling from |psi|^2 using MCMC.
"""

import torch


def compute_local_energy(model, x, V_func, hbar=1.0, mass=1.0):
    """
    Compute local energy E_loc = (H*psi)/psi.

    For H = -hbar^2/(2m) * nabla^2 + V and psi = exp(f):
        E_loc = -hbar^2/(2m) * (laplacian(f) + |grad(f)|^2) + V

    Args:
        model: Network outputting f(x) = log|psi(x)|
        x: Positions, shape (batch, dim)
        V_func: Potential function V(x) -> (batch,)
        hbar: Reduced Planck constant (default 1)
        mass: Particle mass (default 1)

    Returns:
        E_loc: Local energy at each point, shape (batch,)
    """
    batch_size = x.shape[0]
    dim = x.shape[1]

    # Ensure gradients are tracked
    x = x.requires_grad_(True)

    # Compute f = log|psi|
    f = model(x)  # Shape: (batch,)

    # Compute gradient: grad_f[i] = df/dx_i
    grad_f = torch.autograd.grad(
        f.sum(), x,
        create_graph=True,
        retain_graph=True
    )[0]  # Shape: (batch, dim)

    # Compute Laplacian: laplacian(f) = sum_i d^2f/dx_i^2
    laplacian_f = torch.zeros(batch_size, device=x.device)
    for i in range(dim):
        grad_f_i = grad_f[:, i]  # Shape: (batch,)
        grad2_f_i = torch.autograd.grad(
            grad_f_i.sum(), x,
            create_graph=True,
            retain_graph=True
        )[0][:, i]  # d^2f/dx_i^2
        laplacian_f = laplacian_f + grad2_f_i

    # |grad(f)|^2 = sum_i (df/dx_i)^2
    grad_f_sq = torch.sum(grad_f ** 2, dim=-1)  # Shape: (batch,)

    # Kinetic energy: -hbar^2/(2m) * (laplacian(f) + |grad(f)|^2)
    kinetic_coeff = hbar ** 2 / (2.0 * mass)
    kinetic = -kinetic_coeff * (laplacian_f + grad_f_sq)

    # Potential energy
    potential = V_func(x)  # Shape: (batch,)

    return kinetic + potential


def compute_local_energy_and_grad_f(model, x, V_func, hbar=1.0, mass=1.0):
    """
    Compute local energy and grad(f) together for efficiency.

    This is useful for the REINFORCE gradient estimator which needs both.

    Args:
        model: Network outputting f(x) = log|psi(x)|
        x: Positions, shape (batch, dim)
        V_func: Potential function
        hbar: Reduced Planck constant
        mass: Particle mass

    Returns:
        E_loc: Local energy, shape (batch,)
        f: Log amplitude, shape (batch,)
        grad_f: Gradient of f, shape (batch, dim)
    """
    batch_size = x.shape[0]
    dim = x.shape[1]

    x = x.requires_grad_(True)
    f = model(x)

    grad_f = torch.autograd.grad(
        f.sum(), x,
        create_graph=True,
        retain_graph=True
    )[0]

    laplacian_f = torch.zeros(batch_size, device=x.device)
    for i in range(dim):
        grad_f_i = grad_f[:, i]
        grad2_f_i = torch.autograd.grad(
            grad_f_i.sum(), x,
            create_graph=True,
            retain_graph=True
        )[0][:, i]
        laplacian_f = laplacian_f + grad2_f_i

    grad_f_sq = torch.sum(grad_f ** 2, dim=-1)

    kinetic_coeff = hbar ** 2 / (2.0 * mass)
    kinetic = -kinetic_coeff * (laplacian_f + grad_f_sq)
    potential = V_func(x)

    E_loc = kinetic + potential

    return E_loc, f, grad_f


def verify_local_energy_exact(model, x, V_func, E_exact, tol=1e-3):
    """
    Verify that local energy equals exact energy everywhere (for exact psi).

    For the true ground state, E_loc(x) = E_ground for all x.
    This is a useful sanity check.

    Args:
        model: Network that should represent exact ground state
        x: Test positions, shape (batch, dim)
        V_func: Potential function
        E_exact: Known exact energy
        tol: Tolerance for verification

    Returns:
        passed: Whether verification passed
        max_error: Maximum |E_loc - E_exact|
        mean_error: Mean |E_loc - E_exact|
    """
    with torch.no_grad():
        E_loc = compute_local_energy(model, x, V_func)

    errors = torch.abs(E_loc - E_exact)
    max_error = errors.max().item()
    mean_error = errors.mean().item()
    passed = max_error < tol

    return passed, max_error, mean_error


class ExactHarmonicOscillatorModel:
    """
    Exact ground state wavefunction for N-D harmonic oscillator.

    psi(x) = exp(-|x|^2 / 2) / normalization
    f(x) = log|psi| = -|x|^2 / 2 + const

    This class mimics the network interface for testing.
    """

    def __init__(self, dim=1, omega=1.0, hbar=1.0, mass=1.0):
        self.dim = dim
        self.omega = omega
        self.hbar = hbar
        self.mass = mass

        # For harmonic oscillator: psi = exp(-alpha * |x|^2 / 2)
        # where alpha = m*omega/hbar
        self.alpha = mass * omega / hbar

    def __call__(self, x):
        """Compute log|psi(x)| = -alpha/2 * |x|^2."""
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        return -0.5 * self.alpha * torch.sum(x ** 2, dim=-1)

    def parameters(self):
        """Dummy parameters method for compatibility."""
        return iter([])
