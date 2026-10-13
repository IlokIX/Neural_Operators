import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as spla

def evaluate_source(coefficients: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    modes = torch.arange(1, coefficients.shape[-1] + 1, device=coefficients.device, dtype=coefficients.dtype)
    if x.ndim == 1:
        basis = torch.sin(math.pi * x[:, None] * modes[None, :])
        return coefficients @ basis.T
    if x.ndim == 3:
        return torch.sum(coefficients[:, None, :] * torch.sin(math.pi * x * modes[None, None, :]), dim=-1)

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
    def __init__(self, n_sensors, n_states=1, latent_dim=64):
        super().__init__()
        self.n_states = n_states
        self.latent_dim = latent_dim
        
        # Branch net generates now n_states * latent_dim
        self.branch = MLP(input_dim=n_sensors, output_dim=n_states * latent_dim, width=128, depth=3)
        self.trunk = MLP(input_dim=1, output_dim=latent_dim, width=128, depth=3)
        # Independent bias for each state to allow for energy level shifts
        self.bias = nn.Parameter(torch.zeros(n_states))

    def forward(self, f_sensors, x):
        B = f_sensors.shape[0]
        branch_out = self.branch(f_sensors)  # [B, N * P]
        branch_features = branch_out.view(B, self.n_states, self.latent_dim)  # [B, N, P]
        trunk_features = self.trunk(x)       # [B, Q, P]
        
        # Tensor contraction to combine branch and trunk features
        u = torch.einsum("bnp,bqp->bqn", branch_features, trunk_features)
        return u + self.bias

def loss_functions(model, V_sensors, x_collocation, V_collocation):
    x = x_collocation.detach().clone().requires_grad_(True)
    x_sq = x.squeeze(-1) 
    
    raw_out = model(V_sensors, x) # [B, Q, N]
    
    # Ansatz for boundary conditions: psi(x) = x*(1-x)*NN_output
    ansatz_factor = (x_sq * (1.0 - x_sq)).unsqueeze(-1) # [B, Q, 1]
    psi = ansatz_factor * raw_out # [B, Q, N]
    
    B, Q, N = psi.shape
    loss_pde = torch.tensor(0.0, device=x.device)
    loss_norm = torch.tensor(0.0, device=x.device)
    loss_ortho = torch.tensor(0.0, device=x.device)
    energies = []

    for i in range(N):
        psi_i = psi[..., i] # [B, Q]
        
        psi_x = torch.autograd.grad(psi_i, x, grad_outputs=torch.ones_like(psi_i), create_graph=True)[0]
        psi_xx = torch.autograd.grad(psi_x, x, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0].squeeze(-1)

        # Defining the Hamiltonian operator H = -0.5 * d^2/dx^2 + V(x)
        H_psi_i = -0.5 * psi_xx + (V_collocation * psi_i)

        # Energy estimation using the Rayleigh quotient
        E_num = torch.mean(psi_i * H_psi_i, dim=1) 
        E_den = torch.mean(psi_i**2, dim=1)      
        E_i = E_num / (E_den + 1e-8)
        energies.append(E_i)

        # Calculating separate losses
        residual = H_psi_i - (E_i.unsqueeze(1) * psi_i)
        loss_pde += torch.mean(residual.square())
        loss_norm += torch.mean((E_den - 1.0)**2)
        
        # Orthogonality loss: compute overlaps with previous states
        for j in range(i):
            psi_j = psi[..., j]
            overlap = torch.mean(psi_i * psi_j, dim=1)
            loss_ortho += torch.mean(overlap**2)
            
    # Averaging the losses over the number of states
    loss_pde /= N
    loss_norm /= N
    if N > 1:
        loss_ortho /= (N * (N - 1) / 2) # dzielimy przez liczbę par

    energies_tensor = torch.stack(energies, dim=-1) # [B, N]
    return loss_pde, loss_norm, loss_ortho, energies_tensor

def solve_schrodinger_fdm(x_np, V_np, n_states=1):
    N = len(x_np)
    dx = x_np[1] - x_np[0]
    V_int = V_np[1:-1]
    N_int = N - 2
    
    diag_T = np.full(N_int, 1.0 / (dx**2))
    off_diag_T = np.full(N_int - 1, -0.5 / (dx**2))
    diag_H = diag_T + V_int

    evals, evecs = spla.eigh_tridiagonal(diag_H, off_diag_T)

    E_exact = evals[:n_states]
    psi_exact = np.zeros((n_states, N))

    for i in range(n_states):
        psi_int = evecs[:, i]
        psi_exact[i, 1:-1] = psi_int
        norm_factor = np.trapezoid(psi_exact[i]**2, x=x_np)
        psi_exact[i] = psi_exact[i] / np.sqrt(norm_factor)
    
    return E_exact, psi_exact

# Set global plotting parameters
plt.rcParams.update({
    'font.size': 12, 'font.family': 'serif', 'mathtext.fontset': 'cm',
    'axes.labelsize': 14, 'axes.titlesize': 16, 'legend.fontsize': 11
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
    plt.close()

def plot_quantum_comparison(x, V, psi_preds, E_preds, psi_exacts, E_exacts, n_states):
    x_np = x.detach().cpu().numpy()
    V_np = V.detach().cpu().numpy()
    
    fig, axes = plt.subplots(n_states, 2, figsize=(14, 4 * n_states))
    if n_states == 1:
        axes = np.expand_dims(axes, axis=0)
        
    v_range = np.max(V_np) - np.min(V_np)
    scale = (v_range * 0.5) / np.max(np.abs(psi_exacts[0])) if v_range > 0 else 2.0
    
    for i in range(n_states):
        psi_pred_np = psi_preds[i].detach().cpu().numpy()
        E_pred_val = E_preds[i].detach().cpu().numpy().item()
        psi_exact = psi_exacts[i]
        E_exact = E_exacts[i]
        
        # Phase correction
        if np.sum(psi_pred_np * psi_exact) < 0:
            psi_exact = -psi_exact
            
        for ax, is_density, ylabel in zip(axes[i], [False, True], [r"$\psi(x)$", r"$|\psi(x)|^2$"]):
            ax.plot(x_np, V_np, label="Potential $V(x)$", color="violet", linewidth=2, alpha=0.7)
            ax.set_xlabel("x")
            ax.set_ylabel("Energy")
            ax.set_title(f"State $n={i}$ | {ylabel}")
            
            # FDM (exact)
            ax.axhline(E_exact, color="tab:green", linestyle="-", alpha=0.3)
            y_exact = E_exact + scale * (psi_exact**2 if is_density else psi_exact)
            ax.plot(x_np, y_exact, label=f"FDM: E={E_exact:.3f}", color="tab:green", linewidth=3, alpha=0.6)
            
            # DeepONet 
            ax.axhline(E_pred_val, color="tab:blue", linestyle="--", alpha=0.5)
            y_pred = E_pred_val + scale * (psi_pred_np**2 if is_density else psi_pred_np)
            ax.plot(x_np, y_pred, label=f"DeepONet: E={E_pred_val:.3f}", color="tab:blue", linestyle="--", linewidth=2)
            
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            
    plt.suptitle("DeepONet vs Numerical Solution (FDM) for N Energy Levels", y=1.02)
    plt.tight_layout()
    plt.savefig("quantum_comparison.png", bbox_inches='tight')
    plt.close()