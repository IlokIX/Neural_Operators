import torch
import numpy as np
import utils

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_sensors = 50
    latent_dim = 64
    
    sensor_x = torch.linspace(0.0, 1.0, n_sensors).to(device)
    
    model = utils.DeepONet(n_sensors=n_sensors, latent_dim=latent_dim).to(device)\
    
    # Load the model weights
    try:
        model.load_state_dict(torch.load("deeponet_quantum.pth", map_location=device, weights_only=True))
        print("Model loaded successfully from 'deeponet_quantum.pth'")
    except FileNotFoundError:
        print("Error: Model not found. Please ensure 'deeponet_quantum.pth' is in the current directory.")
        return

    model.eval()

    # Generate test coefficients for potentialand evaluate the source term at sensor locations
    test_coefficients = torch.randn(1, 5, device=device)
    V_test_sensors = utils.evaluate_source(test_coefficients, sensor_x)
    
    test_x = torch.linspace(0.0, 1.0, 500, device=device).view(1, 500, 1)
    test_x.requires_grad_(True)
    
    V_test_grid = utils.evaluate_source(test_coefficients, test_x)
    
    raw_out = model(V_test_sensors, test_x)
    
    # Ansatz (boundary conditions): psi(x) = x*(1-x)*NN_output
    test_x_sq = test_x.squeeze(-1) # Size: [1, 500]
    psi_pred = test_x_sq * (1.0 - test_x_sq) * raw_out # Size: [1, 500]
    
    # Normalization with trapezoid method
    norm_factor = torch.trapezoid(psi_pred**2, x=test_x_sq, dim=1) # [1]
    psi_n = psi_pred / torch.sqrt(norm_factor).unsqueeze(1) # [1, 500]
    
    # Evaluation of energy
    psi_x = torch.autograd.grad(psi_n, test_x, grad_outputs=torch.ones_like(psi_n), create_graph=True)[0]
    psi_xx = torch.autograd.grad(psi_x, test_x, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0].squeeze(-1)
    
    H_psi = -0.5 * psi_xx + (V_test_grid.squeeze(-1) * psi_n)
    
    # E_0 = int(psi * H_psi) (denomiantor is already normalized)
    E_pred = torch.trapezoid(psi_n * H_psi, x=test_x_sq, dim=1) # [1]

    #----------------------------------
    # Numerical solution for comparison
    x_numpy = test_x_sq[0].detach().cpu().numpy()
    V_numpy = V_test_grid.squeeze(-1)[0].detach().cpu().numpy()
    
    E_exact, psi_exact = utils.solve_schrodinger_fdm(x_numpy, V_numpy)
    
    # Computing relative errors
    psi_pred_np = psi_n[0].detach().cpu().numpy()
    if np.sum(psi_pred_np * psi_exact) < 0:
        psi_pred_np = -psi_pred_np 
        
    error_l2 = np.linalg.norm(psi_pred_np - psi_exact) / np.linalg.norm(psi_exact)
    error_E = abs(E_pred[0].item() - E_exact) / abs(E_exact)
    
    print(f"Relative L2 error for psi(x): {error_l2:.4e}")
    print(f"Relative L2 energy error E: {error_E:.4e}")
    
    # Plottitng the results
    utils.plot_quantum_comparison(
        x=test_x_sq[0], 
        V=V_test_grid.squeeze(-1)[0], 
        psi_pred=psi_n[0],
        E_pred=E_pred[0],
        psi_exact=psi_exact,
        E_exact=E_exact
    )

if __name__ == "__main__":
    main()