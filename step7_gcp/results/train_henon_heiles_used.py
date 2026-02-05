#!/usr/bin/env python3
"""
Self-contained LP+PINN training script for 6D Hénon-Heiles system.

This script is designed to run on GCP with no dependencies on other project code.
It implements the Lagrangian PINN approach for quantum ground state calculation.

Network Architecture:
- 8 hidden layers x 512 neurons
- Tanh activation
- Input: 6 (position coordinates)
- Output: 12 (real + imaginary momentum components)

Loss Functions:
- Physics loss: QHJE residual (p² + iℏ∇·p = 2m(E-V))
- Curl loss: Irrotationality (∂pᵢ/∂xⱼ = ∂pⱼ/∂xᵢ)

Hénon-Heiles Potential (6D with periodic coupling):
V(x) = ½Σᵢxᵢ² + λΣᵢ(xᵢ²xᵢ₊₁ - xᵢ₊₁³/3)
where x₇ = x₁ (periodic) and λ = 1/√80 ≈ 0.111803
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


#=============================================================================
# CONFIGURATION
#=============================================================================

class Config:
    """Training configuration."""
    # Network architecture
    input_dim = 6
    output_dim = 12  # 6 real + 6 imaginary momentum components
    hidden_layers = 8
    hidden_dim = 512
    activation = "tanh"

    # Physics parameters
    hbar = 1.0  # Natural units
    mass = 1.0
    lambda_hh = 1.0 / np.sqrt(80)  # ≈ 0.111803

    # Training parameters
    learning_rate = 1e-4
    n_collocation = 16_000  # Reduced for memory
    max_epochs = 50_000
    checkpoint_interval = 1000
    log_interval = 100

    # Domain (sampling range for each coordinate)
    domain_min = -3.0
    domain_max = 3.0

    # Loss weights
    weight_physics = 1.0
    weight_curl = 0.1

    # Initial energy guess (harmonic ZPE = n*hbar*omega/2 = 6*1*1/2 = 3.0)
    initial_energy = 3.0


#=============================================================================
# NETWORK ARCHITECTURE
#=============================================================================

class LagrangianPINN(nn.Module):
    """
    Lagrangian Physics-Informed Neural Network for quantum mechanics.

    Outputs complex momentum field p(x) = p_real(x) + i*p_imag(x)
    for the quantum Hamilton-Jacobi equation.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Build network layers
        layers = []

        # Input layer
        layers.append(nn.Linear(config.input_dim, config.hidden_dim))
        layers.append(nn.Tanh())

        # Hidden layers
        for _ in range(config.hidden_layers - 1):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            layers.append(nn.Tanh())

        # Output layer
        layers.append(nn.Linear(config.hidden_dim, config.output_dim))

        self.network = nn.Sequential(*layers)

        # Learnable energy parameter
        self.log_energy = nn.Parameter(torch.tensor(np.log(config.initial_energy)))

        # Initialize weights using Xavier
        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better gradient flow."""
        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @property
    def energy(self):
        """Current energy estimate (always positive via exp)."""
        return torch.exp(self.log_energy)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Position tensor of shape (batch, 6)

        Returns:
            p_real: Real momentum components (batch, 6)
            p_imag: Imaginary momentum components (batch, 6)
        """
        output = self.network(x)
        p_real = output[:, :6]
        p_imag = output[:, 6:]
        return p_real, p_imag


#=============================================================================
# PHYSICS
#=============================================================================

def henon_heiles_potential(x, lambda_hh):
    """
    6D Hénon-Heiles potential with periodic coupling.

    V(x) = ½Σᵢxᵢ² + λΣᵢ(xᵢ²xᵢ₊₁ - xᵢ₊₁³/3)

    Args:
        x: Position tensor of shape (batch, 6)
        lambda_hh: Coupling constant

    Returns:
        V: Potential tensor of shape (batch,)
    """
    # Harmonic term: ½Σxᵢ²
    V_harmonic = 0.5 * torch.sum(x**2, dim=1)

    # Anharmonic term with periodic boundary (x₇ = x₁)
    V_anharmonic = torch.zeros_like(V_harmonic)
    for i in range(6):
        j = (i + 1) % 6  # Periodic: x₇ = x₁
        xi = x[:, i]
        xj = x[:, j]
        V_anharmonic += xi**2 * xj - xj**3 / 3

    return V_harmonic + lambda_hh * V_anharmonic


def compute_gradient(y, x):
    """
    Compute gradient ∇y with respect to x.

    Args:
        y: Scalar field tensor of shape (batch,)
        x: Position tensor of shape (batch, dim)

    Returns:
        grad_y: Gradient tensor of shape (batch, dim)
    """
    grad_y = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True
    )[0]
    return grad_y


def compute_divergence(p, x):
    """
    Compute divergence ∇·p.

    Args:
        p: Vector field tensor of shape (batch, dim)
        x: Position tensor of shape (batch, dim)

    Returns:
        div_p: Divergence tensor of shape (batch,)
    """
    div_p = torch.zeros(x.shape[0], device=x.device)
    for i in range(x.shape[1]):
        grad_pi = torch.autograd.grad(
            outputs=p[:, i],
            inputs=x,
            grad_outputs=torch.ones_like(p[:, i]),
            create_graph=True,
            retain_graph=True
        )[0]
        div_p += grad_pi[:, i]
    return div_p


def compute_curl_loss(p_real, p_imag, x):
    """
    Compute curl loss for irrotationality constraint.

    For a gradient field, ∂pᵢ/∂xⱼ = ∂pⱼ/∂xᵢ.

    Args:
        p_real: Real momentum (batch, dim)
        p_imag: Imaginary momentum (batch, dim)
        x: Position (batch, dim)

    Returns:
        curl_loss: Scalar loss
    """
    dim = x.shape[1]
    curl_loss = torch.tensor(0.0, device=x.device)

    # Compute all partial derivatives for real part
    grad_p_real = []
    for i in range(dim):
        grad_pi = torch.autograd.grad(
            outputs=p_real[:, i],
            inputs=x,
            grad_outputs=torch.ones_like(p_real[:, i]),
            create_graph=True,
            retain_graph=True
        )[0]
        grad_p_real.append(grad_pi)

    # Check symmetry: ∂pᵢ/∂xⱼ = ∂pⱼ/∂xᵢ
    for i in range(dim):
        for j in range(i + 1, dim):
            diff = grad_p_real[i][:, j] - grad_p_real[j][:, i]
            curl_loss += torch.mean(diff**2)

    # Same for imaginary part
    grad_p_imag = []
    for i in range(dim):
        grad_pi = torch.autograd.grad(
            outputs=p_imag[:, i],
            inputs=x,
            grad_outputs=torch.ones_like(p_imag[:, i]),
            create_graph=True,
            retain_graph=True
        )[0]
        grad_p_imag.append(grad_pi)

    for i in range(dim):
        for j in range(i + 1, dim):
            diff = grad_p_imag[i][:, j] - grad_p_imag[j][:, i]
            curl_loss += torch.mean(diff**2)

    return curl_loss


def compute_physics_loss(model, x, config):
    """
    Compute QHJE physics loss.

    Quantum Hamilton-Jacobi equation:
    (p_real² - p_imag²)/2m + V = E  (real part)
    p_real·p_imag/m + (ℏ/2m)∇·p_real = 0  (imaginary part, simplified)

    With quantum potential correction.

    Args:
        model: LagrangianPINN model
        x: Collocation points (batch, dim)
        config: Training configuration

    Returns:
        physics_loss: Scalar loss
        energy: Current energy estimate
    """
    x.requires_grad_(True)

    # Get momentum field
    p_real, p_imag = model(x)

    # Compute potential
    V = henon_heiles_potential(x, config.lambda_hh)

    # Compute divergences
    div_p_real = compute_divergence(p_real, x)
    div_p_imag = compute_divergence(p_imag, x)

    # Kinetic terms
    p_real_sq = torch.sum(p_real**2, dim=1)
    p_imag_sq = torch.sum(p_imag**2, dim=1)
    p_cross = torch.sum(p_real * p_imag, dim=1)

    # Energy
    E = model.energy

    # QHJE residuals (with mass = 1, hbar = 1)
    # Real part: (p_r² - p_i²)/2 + V - (ℏ/2)∇·p_i = E
    real_residual = (p_real_sq - p_imag_sq) / 2 + V - 0.5 * config.hbar * div_p_imag - E

    # Imaginary part: p_r·p_i + (ℏ/2)∇·p_r = 0
    imag_residual = p_cross + 0.5 * config.hbar * div_p_real

    # Combined physics loss
    physics_loss = torch.mean(real_residual**2) + torch.mean(imag_residual**2)

    return physics_loss, E


#=============================================================================
# TRAINING
#=============================================================================

def sample_collocation_points(n_points, config, device):
    """
    Sample collocation points uniformly in the domain.

    Uses importance sampling biased toward lower potential regions.
    """
    # Uniform sampling in domain
    x = torch.rand(n_points, config.input_dim, device=device)
    x = x * (config.domain_max - config.domain_min) + config.domain_min
    return x


def train_epoch(model, optimizer, config, device):
    """
    Train for one epoch.

    Returns:
        loss: Total loss
        physics_loss: Physics component
        curl_loss: Curl component
        energy: Current energy estimate
    """
    model.train()
    optimizer.zero_grad()

    # Sample collocation points
    x = sample_collocation_points(config.n_collocation, config, device)
    x.requires_grad_(True)

    # Compute physics loss
    physics_loss, energy = compute_physics_loss(model, x, config)

    # Compute curl loss
    p_real, p_imag = model(x)
    curl_loss = compute_curl_loss(p_real, p_imag, x)

    # Total loss
    loss = config.weight_physics * physics_loss + config.weight_curl * curl_loss

    # Backward pass
    loss.backward()

    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    return (
        loss.item(),
        physics_loss.item(),
        curl_loss.item(),
        energy.item()
    )


def save_checkpoint(model, optimizer, epoch, loss, energy, checkpoint_dir):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'energy': energy,
        'config': {
            'hidden_layers': model.config.hidden_layers,
            'hidden_dim': model.config.hidden_dim,
            'lambda_hh': model.config.lambda_hh,
        }
    }

    path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch:06d}.pt"
    torch.save(checkpoint, path)

    # Also save as 'latest' for easy resume
    latest_path = Path(checkpoint_dir) / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)

    return path


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint.get('loss', 0), checkpoint.get('energy', 3.0)


def save_results(results, results_dir):
    """Save final results to JSON."""
    path = Path(results_dir) / "final_results.json"
    with open(path, 'w') as f:
        json.dump(results, f, indent=2)
    return path


#=============================================================================
# MAIN
#=============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train LP+PINN for 6D Hénon-Heiles')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory for checkpoints')
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory for results')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs')
    args = parser.parse_args()

    # Setup
    config = Config()
    if args.epochs:
        config.max_epochs = args.epochs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create directories
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # Initialize model
    model = LagrangianPINN(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_epoch, last_loss, last_energy = load_checkpoint(
            model, optimizer, args.resume, device
        )
        print(f"Resumed at epoch {start_epoch}, loss={last_loss:.6f}, energy={last_energy:.6f}")
        start_epoch += 1  # Start from next epoch

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training: epochs {start_epoch} to {config.max_epochs}")
    print(f"Collocation points: {config.n_collocation:,}")
    print(f"Checkpoint interval: {config.checkpoint_interval}")
    print(f"{'='*60}\n")

    best_energy = float('inf')
    training_start = time.time()

    history = {
        'epochs': [],
        'loss': [],
        'physics_loss': [],
        'curl_loss': [],
        'energy': []
    }

    try:
        for epoch in range(start_epoch, config.max_epochs):
            epoch_start = time.time()

            loss, physics_loss, curl_loss, energy = train_epoch(
                model, optimizer, config, device
            )

            epoch_time = time.time() - epoch_start

            # Track history
            history['epochs'].append(epoch)
            history['loss'].append(loss)
            history['physics_loss'].append(physics_loss)
            history['curl_loss'].append(curl_loss)
            history['energy'].append(energy)

            # Track best energy
            if energy < best_energy:
                best_energy = energy

            # Logging
            if epoch % config.log_interval == 0:
                elapsed = time.time() - training_start
                print(f"Epoch {epoch:6d} | "
                      f"Loss: {loss:.6f} | "
                      f"Physics: {physics_loss:.6f} | "
                      f"Curl: {curl_loss:.6f} | "
                      f"Energy: {energy:.6f} | "
                      f"Time: {epoch_time:.2f}s | "
                      f"Elapsed: {elapsed/3600:.2f}h")

            # Checkpointing
            if epoch > 0 and epoch % config.checkpoint_interval == 0:
                ckpt_path = save_checkpoint(
                    model, optimizer, epoch, loss, energy, args.checkpoint_dir
                )
                print(f"  -> Saved checkpoint: {ckpt_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Final results
    total_time = time.time() - training_start
    final_energy = model.energy.item()

    # Reference values
    harmonic_zpe = 3.0  # 6D harmonic oscillator ZPE
    expected_energy = 2.97  # Approximate anharmonic ground state

    results = {
        'final_energy': final_energy,
        'best_energy': best_energy,
        'harmonic_zpe': harmonic_zpe,
        'expected_energy': expected_energy,
        'below_harmonic_zpe': final_energy < harmonic_zpe,
        'error_vs_expected': abs(final_energy - expected_energy) / expected_energy * 100,
        'total_epochs': len(history['epochs']),
        'training_time_hours': total_time / 3600,
        'final_loss': history['loss'][-1] if history['loss'] else None,
        'config': {
            'hidden_layers': config.hidden_layers,
            'hidden_dim': config.hidden_dim,
            'n_collocation': config.n_collocation,
            'learning_rate': config.learning_rate,
            'lambda_hh': config.lambda_hh,
        },
        'timestamp': datetime.now().isoformat()
    }

    # Save results
    results_path = save_results(results, args.results_dir)

    # Save final checkpoint
    final_ckpt = save_checkpoint(
        model, optimizer, len(history['epochs']),
        history['loss'][-1] if history['loss'] else 0,
        final_energy, args.checkpoint_dir
    )

    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Final energy:      {final_energy:.6f} a.u.")
    print(f"Best energy:       {best_energy:.6f} a.u.")
    print(f"Harmonic ZPE:      {harmonic_zpe:.6f} a.u.")
    print(f"Expected (lit):    {expected_energy:.6f} a.u.")
    print(f"Below harmonic:    {'YES' if final_energy < harmonic_zpe else 'NO'}")
    print(f"Error vs expected: {results['error_vs_expected']:.2f}%")
    print(f"Training time:     {total_time/3600:.2f} hours")
    print(f"Results saved:     {results_path}")
    print(f"Final checkpoint:  {final_ckpt}")
    print(f"{'='*60}")

    # Success criteria check
    if 2.91 <= final_energy <= 3.03:
        print("\n✓ SUCCESS: Energy within 2% of target (2.97 ± 2%)")
    else:
        print(f"\n✗ Target not met: {final_energy:.4f} outside [2.91, 3.03]")


if __name__ == '__main__':
    main()
