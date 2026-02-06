"""
MCMC samplers for variational Monte Carlo.

The key challenge is sampling from |psi(x)|^2 where psi is defined
by a neural network. We use Metropolis-Hastings with Gaussian proposals.
"""

import torch
import numpy as np


class MetropolisSampler:
    """
    Metropolis-Hastings sampler for |psi|^2 distribution.

    Uses Gaussian proposal distribution with adaptive step size.

    Args:
        model: Network that outputs log|psi(x)|
        dim: Number of spatial dimensions
        n_walkers: Number of parallel walkers (samples per batch)
        step_size: Initial Gaussian proposal standard deviation
        device: torch device
    """

    def __init__(self, model, dim, n_walkers=1000, step_size=0.5, device=None):
        self.model = model
        self.dim = dim
        self.n_walkers = n_walkers
        self.step_size = step_size

        if device is None:
            device = next(model.parameters()).device
        self.device = device

        # Initialize walkers from standard Gaussian
        # This is reasonable for harmonic oscillator ground state
        self.walkers = torch.randn(n_walkers, dim, device=device)

        # Track acceptance rate for diagnostics
        self.acceptance_rate = 0.5

    def sample(self, n_steps=10, burn_in=0, adapt_step=False):
        """
        Run MCMC for n_steps, return final walker positions.

        Args:
            n_steps: Number of MCMC steps to run
            burn_in: Additional burn-in steps (discarded)
            adapt_step: If True, adapt step_size to target 50% acceptance

        Returns:
            samples: Walker positions, shape (n_walkers, dim)
        """
        total_steps = burn_in + n_steps
        n_accepted = 0
        n_proposed = 0

        with torch.no_grad():
            for step in range(total_steps):
                # Propose moves: x' = x + step_size * N(0, 1)
                proposals = self.walkers + self.step_size * torch.randn_like(self.walkers)

                # Compute log|psi|^2 = 2*f for current and proposed
                log_prob_current = 2.0 * self.model(self.walkers)
                log_prob_proposed = 2.0 * self.model(proposals)

                # Acceptance probability: min(1, |psi'|^2 / |psi|^2)
                # In log space: log(accept_prob) = log|psi'|^2 - log|psi|^2
                log_accept_prob = log_prob_proposed - log_prob_current

                # Accept if log(U) < log(accept_prob) where U ~ Uniform(0,1)
                log_uniform = torch.log(torch.rand(self.n_walkers, device=self.device))
                accept = log_uniform < log_accept_prob

                # Update accepted walkers
                self.walkers = torch.where(
                    accept.unsqueeze(-1),
                    proposals,
                    self.walkers
                )

                # Track acceptance rate (only after burn-in)
                if step >= burn_in:
                    n_accepted += accept.sum().item()
                    n_proposed += self.n_walkers

                # Adapt step size during burn-in
                if adapt_step and step < burn_in and step > 0 and step % 10 == 0:
                    current_rate = accept.float().mean().item()
                    if current_rate > 0.6:
                        self.step_size *= 1.1
                    elif current_rate < 0.4:
                        self.step_size *= 0.9

        # Update acceptance rate
        if n_proposed > 0:
            self.acceptance_rate = n_accepted / n_proposed

        return self.walkers.clone()

    def reset(self, init_scale=1.0):
        """Reset walkers to random positions."""
        self.walkers = init_scale * torch.randn(
            self.n_walkers, self.dim, device=self.device
        )

    def thermalize(self, n_steps=100):
        """Run burn-in without returning samples."""
        self.sample(n_steps=0, burn_in=n_steps, adapt_step=True)


class HamiltonianMonteCarlo:
    """
    Hamiltonian Monte Carlo (HMC) sampler for better mixing in high dimensions.

    HMC uses gradient information to propose moves that are more likely
    to be accepted, improving efficiency in high-dimensional spaces.

    Note: Requires gradients of log|psi|, which means model evaluation
    must be done with gradients enabled.

    Args:
        model: Network that outputs log|psi(x)|
        dim: Number of spatial dimensions
        n_walkers: Number of parallel walkers
        step_size: Leapfrog integration step size
        n_leapfrog: Number of leapfrog steps per proposal
        device: torch device
        V_func: Optional potential function for V<0 rejection (bounded sampling)
        target_acceptance: Target acceptance rate for step size adaptation (default 0.65)
    """

    def __init__(self, model, dim, n_walkers=1000, step_size=0.1,
                 n_leapfrog=10, device=None, V_func=None, target_acceptance=0.65):
        self.model = model
        self.dim = dim
        self.n_walkers = n_walkers
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog

        if device is None:
            device = next(model.parameters()).device
        self.device = device

        self.walkers = torch.randn(n_walkers, dim, device=device)
        self.acceptance_rate = 0.5
        self.V_func = V_func
        self.target_acceptance = target_acceptance
        self.v_rejection_rate = 0.0

    def _grad_log_prob(self, x):
        """Compute gradient of log|psi|^2 = 2*grad(f)."""
        x = x.requires_grad_(True)
        log_prob = 2.0 * self.model(x)
        grad = torch.autograd.grad(log_prob.sum(), x)[0]
        return grad.detach()

    def sample(self, n_steps=10, burn_in=0):
        """
        Run HMC for n_steps.

        Args:
            n_steps: Number of HMC steps
            burn_in: Burn-in steps (discarded)

        Returns:
            samples: Walker positions, shape (n_walkers, dim)
        """
        total_steps = burn_in + n_steps
        n_accepted = 0
        n_v_rejected = 0
        n_total = 0

        for step in range(total_steps):
            # Sample momentum from N(0, 1)
            p = torch.randn_like(self.walkers)

            # Current position and Hamiltonian
            q = self.walkers.clone()
            with torch.no_grad():
                log_prob_q = 2.0 * self.model(q)
            H_current = -log_prob_q + 0.5 * (p ** 2).sum(dim=-1)

            # Leapfrog integration
            q_new = q.clone()
            p_new = p.clone()

            # Half step for momentum
            grad = self._grad_log_prob(q_new)
            p_new = p_new + 0.5 * self.step_size * grad

            # Full steps
            for _ in range(self.n_leapfrog - 1):
                q_new = q_new + self.step_size * p_new
                grad = self._grad_log_prob(q_new)
                p_new = p_new + self.step_size * grad

            # Final position step and half momentum step
            q_new = q_new + self.step_size * p_new
            grad = self._grad_log_prob(q_new)
            p_new = p_new + 0.5 * self.step_size * grad

            # Proposed Hamiltonian
            with torch.no_grad():
                log_prob_q_new = 2.0 * self.model(q_new)
            H_proposed = -log_prob_q_new + 0.5 * (p_new ** 2).sum(dim=-1)

            # Check V<0 constraint if V_func provided
            if self.V_func is not None:
                with torch.no_grad():
                    V_new = self.V_func(q_new)
                    v_valid = V_new >= 0
            else:
                v_valid = torch.ones(self.n_walkers, dtype=torch.bool, device=self.device)

            # Accept/reject: Metropolis AND V>=0
            log_accept_prob = H_current - H_proposed
            log_uniform = torch.log(torch.rand(self.n_walkers, device=self.device))
            metropolis_accept = log_uniform < log_accept_prob
            accept = v_valid & metropolis_accept

            self.walkers = torch.where(
                accept.unsqueeze(-1),
                q_new,
                self.walkers
            )

            if step >= burn_in:
                n_accepted += accept.sum().item()
                n_v_rejected += (~v_valid).sum().item()
                n_total += self.n_walkers

        if n_total > 0:
            self.acceptance_rate = n_accepted / n_total
            self.v_rejection_rate = n_v_rejected / n_total

        return self.walkers.clone()

    def adapt_step_size(self):
        """Adjust step size to target acceptance rate."""
        if self.acceptance_rate < self.target_acceptance - 0.1:
            self.step_size *= 0.8  # Decrease faster when too low
        elif self.acceptance_rate > self.target_acceptance + 0.1:
            self.step_size *= 1.2  # Increase faster when too high
        # Clamp to reasonable range (allow larger steps for high-D)
        self.step_size = max(0.001, min(2.0, self.step_size))

    def thermalize(self, n_steps=100, adapt_every=20):
        """
        Thermalize with step size adaptation.

        Args:
            n_steps: Total thermalization steps
            adapt_every: Adapt step size every N steps
        """
        for i in range(n_steps):
            self.sample(n_steps=1)
            if (i + 1) % adapt_every == 0:
                self.adapt_step_size()

    def reset(self, init_scale=1.0):
        """Reset walkers to random positions."""
        self.walkers = init_scale * torch.randn(
            self.n_walkers, self.dim, device=self.device
        )
