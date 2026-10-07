import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt


#torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Generating problems family
# u''(x) = -f(x), u(0) = u(1) = 0
# f(x) = sum_k a_k sin(k*pi*x)
# u(x) = sum_k a_k/(k*pi)^2 sin(k*pi*x)
# ---------------------------------------------------------
def generate_functions(
    n_functions: int,
    sensor_x: torch.Tensor,
    query_x: torch.Tensor,
    n_modes: int = 5,
):
    """
    Returns:
        f_sensors: values of f at sensors, shape [B, M]
        x_query: query points, shape [B, Q, 1]
        u_query: solution at query points, shape [B, Q]
    """
    coefficients = torch.randn(n_functions, n_modes)

    modes = torch.arange(1, n_modes + 1, dtype=torch.float32)

    # [M, K]
    sensor_basis = torch.sin(
        math.pi * sensor_x[:, None] * modes[None, :]
    )

    # [Q, K]
    query_basis = torch.sin(
        math.pi * query_x[:, None] * modes[None, :]
    )

    # f(x_sensor), shape [B, M]
    f_sensors = coefficients @ sensor_basis.T

    # u(x_query), shape [B, Q]
    denominators = (math.pi * modes) ** 2
    u_query = (coefficients / denominators[None, :]) @ query_basis.T

    x_query = query_x[None, :, None].repeat(n_functions, 1, 1)

    return f_sensors, x_query, u_query


def evaluate_source(coefficients: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Evaluate f(x)=sum_k a_k sin(k*pi*x).

    coefficients: [B, K]
    x: [B, Q, 1]
    returns: [B, Q]
    """
    modes = torch.arange(
        1, coefficients.shape[-1] + 1,
        device=coefficients.device,
        dtype=coefficients.dtype,
    )
    if x.ndim == 1:
        basis = torch.sin(math.pi * x[:, None] * modes[None, :])  # [Q,K]
        return coefficients @ basis.T
    if x.ndim == 3:
        return torch.sum(
            coefficients[:, None, :] * torch.sin(math.pi * x * modes[None, None, :]),
            dim=-1,
        )

def sample_problem_batch(batch_size, sensor_x, n_collocation, n_modes, device):
    """Create one physics-informed batch.

    Returns branch input f(sensor_x), collocation coordinates, and f(x_c).
    No exact u is used for training.
    """
    coefficients = torch.randn(batch_size, n_modes, device=device)
    sensor_x = sensor_x.to(device)
    f_sensors = evaluate_source(coefficients, sensor_x)  # [B,M]

    # Random interior points; independent points for every input function.
    x_collocation = torch.rand(batch_size, n_collocation, 1, device=device)
    f_collocation = evaluate_source(coefficients, x_collocation)  # [B,Q]
    return coefficients, f_sensors, x_collocation, f_collocation


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


#---------------------------------------------------------
# DeepONet:
# [f(x1), f(x2), ..., f(xM)] -branch net-> [b1, b2, ..., bP]
# [x] -trunk net-> [t1, t2, ..., tP]
# uNN(x) = sum_i^P b_i * t_i + bias
# M == number of sensors = n_sensors
#---------------------------------------------------------
class DeepONet(nn.Module):
    def __init__(self, n_sensors, latent_dim=64):
        super().__init__()

        self.branch = MLP(
            input_dim=n_sensors,
            output_dim=latent_dim,
            width=128,
            depth=3,
        )

        self.trunk = MLP(
            input_dim=1,
            output_dim=latent_dim,
            width=128,
            depth=3,
        )

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, f_sensors, x):
        """
        f_sensors: [B, M]
        x:         [B, Q, 1]

        output:    [B, Q]
        """
        branch_features = self.branch(f_sensors)       # [B, P]
        trunk_features = self.trunk(x)                 # [B, Q, P]

        u = torch.einsum(
            "bp,bqp->bq",
            branch_features,
            trunk_features,
        )

        return u + self.bias
    

def compute_quantum_losses_mc(model, V_sensors, x_collocation, V_collocation, psi_prev=None):
    """
    x_collocation: [B, Q, 1] - punkty losowane z torch.rand
    """
    x = x_collocation.detach().clone().requires_grad_(True)
    x_sq = x.squeeze(-1) # [B, Q]
    
    # Hard boundary conditions: psi(0)=psi(1)=0
    psi = x_sq * (1.0 - x_sq) * raw_out
    
    # Graduate psi with respect to x to compute derivatives
    psi_x = torch.autograd.grad(
        psi, x,
        grad_outputs=torch.ones_like(psi),
        create_graph=True,
    )[0]
    
    psi_xx = torch.autograd.grad(
        psi_x, x,
        grad_outputs=torch.ones_like(psi_x),
        create_graph=True,
    )[0].squeeze(-1)
    
    H_psi = -0.5 * psi_xx + (V_collocation * psi)
    
    # Monte Carlo integration to compute expectation values
    E_num = torch.mean(psi * H_psi, dim=1) # [B]
    E_den = torch.mean(psi**2, dim=1)      # [B]
    
    E = E_num / (E_den + 1e-8) # [B]
    
    # Separate the PDE residual for loss computation
    residual = H_psi - (E.unsqueeze(1) * psi)
    loss_pde = torch.mean(residual.square())
    
    # Normalization loss
    loss_norm = torch.mean((E_den - 1.0)**2)
    
    # Orthogonalization loss with respect to previous wavefunctions
    loss_ortho = torch.tensor(0.0, device=x.device)
    if psi_prev is not None:
        overlap = torch.mean(psi * psi_prev, dim=1)
        loss_ortho = torch.mean(overlap**2)
        
    return loss_pde, loss_norm, loss_ortho, E


def plot_predictions(x, u_exact, u_pred, f_source=None):

    plt.figure(figsize=(8, 5))
    plt.plot(x.squeeze(), u_exact.squeeze(), label="Exact", linewidth=2)
    plt.plot(x.squeeze(), u_pred.squeeze(), label="Predicted", linestyle="--")
    if f_source is not None:
        plt.plot(f_source[0].squeeze(), f_source[1].squeeze()/10., label="Source", linestyle=":")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("DeepONet Predictions vs Exact Solution")
    plt.legend()
    plt.grid()
    plt.savefig("predictions.png")
    
def plot_loss(loss_history, no_epochs_per_tick=10):
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(loss_history))*no_epochs_per_tick, loss_history, label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.savefig("loss_history.png")