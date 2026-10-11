import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Data Generation
# ---------------------------------------------------------
def evaluate_source(coefficients: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    modes = torch.arange(
        1, coefficients.shape[-1] + 1,
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    if x.ndim == 1:
        basis = torch.sin(math.pi * x[:, None] * modes[None, :])
        return coefficients @ basis.T
    if x.ndim == 3:
        return torch.sum(
            coefficients[:, None, :] * torch.sin(math.pi * x * modes[None, None, :]),
            dim=-1,
        )

# ---------------------------------------------------------
# Model architekture
# ---------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, width=128, depth=3):
        super().__init__()
        layers = [nn.Linear(input_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class DeepONet(nn.Module):
    def __init__(self, n_sensors, latent_dim=64):
        super().__init__()
        self.branch = MLP(input_dim=n_sensors, output_dim=latent_dim, width=128, depth=3)
        self.trunk = MLP(input_dim=1, output_dim=latent_dim, width=128, depth=3)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, f_sensors, x):
        branch_features = self.branch(f_sensors)
        trunk_features = self.trunk(x)
        u = torch.einsum("bp,bqp->bq", branch_features, trunk_features)
        return u + self.bias

# ---------------------------------------------------------
# Physics-Informed Loss Functions
# ---------------------------------------------------------
def loss_functions(model, V_sensors, x_collocation, V_collocation, psi_prev=None):
    x = x_collocation.detach().clone().requires_grad_(True)
    x_sq = x.squeeze(-1) 
    
    # Ansatz: psi(x) = x*(1-x)*NN_output
    raw_out = model(V_sensors, x)
    psi = x_sq * (1.0 - x_sq) * raw_out
    
    psi_x = torch.autograd.grad(
        psi, x, grad_outputs=torch.ones_like(psi), create_graph=True
    )[0]
    
    psi_xx = torch.autograd.grad(
        psi_x, x, grad_outputs=torch.ones_like(psi_x), create_graph=True
    )[0].squeeze(-1)
    
    # Defining the Hamiltonian operator H = -0.5 * d^2/dx^2 + V(x)
    H_psi = -0.5 * psi_xx + (V_collocation * psi)
    
    # Energy estimation using the Rayleigh quotient
    E_num = torch.mean(psi * H_psi, dim=1) 
    E_den = torch.mean(psi**2, dim=1)      
    E = E_num / (E_den + 1e-8) 
    
    # Calculating separate losses
    residual = H_psi - (E.unsqueeze(1) * psi)
    loss_pde = torch.mean(residual.square())
    loss_norm = torch.mean((E_den - 1.0)**2)
    
    loss_ortho = torch.tensor(0.0, device=x.device)
    if psi_prev is not None:
        overlap = torch.mean(psi * psi_prev, dim=1)
        loss_ortho = torch.mean(overlap**2)
        
    return loss_pde, loss_norm, loss_ortho, E


#---------------------------------------------------------
# FDM solver for comparison
#---------------------------------------------------------

import scipy.linalg as spla

def solve_schrodinger_fdm(x_np, V_np):
    """
    Calculate the ground state energy and wavefunction of the 1D Schrödinger equation
    using the Finite Difference Method (FDM).
    """
    N = len(x_np)
    dx = x_np[1] - x_np[0]
    
    V_int = V_np[1:-1]
    N_int = N - 2
    
    diag_T = np.full(N_int, 1.0 / (dx**2))
    off_diag_T = np.full(N_int - 1, -0.5 / (dx**2))
    
    diag_H = diag_T + V_int

    evals, evecs = spla.eigh_tridiagonal(diag_H, off_diag_T)

    E0_exact = evals[0]
    psi0_int = evecs[:, 0]

    psi0_exact = np.zeros(N)
    psi0_exact[1:-1] = psi0_int

    norm_factor = np.trapezoid(psi0_exact**2, x=x_np)
    psi0_exact = psi0_exact / np.sqrt(norm_factor)
    
    return E0_exact, psi0_exact

# ---------------------------------------------------------
# Visualization of Loss Components
# ---------------------------------------------------------

# Set global plotting parameters for better aesthetics
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11
})

def plot_separated_losses(hist_total, hist_pde, hist_norm, hist_ortho, no_epochs_per_tick=10):
    epochs = np.arange(len(hist_total)) * no_epochs_per_tick
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, hist_total, label="Total Loss", linewidth=3, color='black', alpha=0.7)
    plt.plot(epochs, hist_pde, label="PDE Loss", linewidth=2, linestyle='--')
    plt.plot(epochs, hist_norm, label="Norm Loss", linewidth=2, linestyle='-.')
    
    if any(v > 0 for v in hist_ortho):
        plt.plot(epochs, hist_ortho, label="Ortho Loss", linewidth=2, linestyle=':')
        
    plt.xlabel("Epochs")
    plt.ylabel("Loss Component Value")
    plt.title("Training Loss breakdown: PINN DeepONet")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.savefig("loss_breakdown.png")
    plt.show()


#---------------------------------------------------------
# State comparation and visualization
#---------------------------------------------------------

def plot_quantum_comparison(x, V, psi_pred, E_pred, psi_exact, E_exact):
    x_np = x.detach().cpu().numpy()
    V_np = V.detach().cpu().numpy()
    psi_pred_np = psi_pred.detach().cpu().numpy()
    E_pred_val = E_pred.detach().cpu().numpy().item()
    
    # Ensure the predicted wavefunction has the same sign as the exact one for meaningful comparison
    if np.sum(psi_pred_np * psi_exact) < 0:
        psi_exact = -psi_exact
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    v_range = np.max(V_np) - np.min(V_np)
    scale = (v_range * 0.5) / np.max(np.abs(psi_exact)) if v_range > 0 else 2.0
    
    for ax, is_density, ylabel in zip([ax1, ax2], [False, True], [r"Wave function $\psi(x)$", r"Probability density $|\psi(x)|^2$"]):
        ax.plot(x_np, V_np, label="Potential $V(x)$", color="violet", linewidth=2, alpha=0.7)
        ax.set_xlabel("x (Position)")
        ax.set_ylabel("Energy")
        ax.set_title(ylabel)
        
        # FDM solution (exact)
        ax.axhline(E_exact, color="tab:green", linestyle="-", alpha=0.3)
        y_exact = E_exact + scale * (psi_exact**2 if is_density else psi_exact)
        ax.plot(x_np, y_exact, label=f"FDM: E={E_exact:.3f}", color="tab:green", linewidth=3, alpha=0.6)
        
        # DeepONet prediction
        ax.axhline(E_pred_val, color="tab:blue", linestyle="--", alpha=0.5)
        y_pred = E_pred_val + scale * (psi_pred_np**2 if is_density else psi_pred_np)
        ax.plot(x_np, y_pred, label=f"DeepONet: E={E_pred_val:.3f}", color="tab:blue", linestyle="--", linewidth=2)
        
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        
    plt.suptitle("Comparison of DeepONet predictions with numerical solution (FDM)")
    plt.tight_layout()
    plt.savefig("quantum_comparison.png")
    plt.show()