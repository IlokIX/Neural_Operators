import torch
import numpy as np
import utils

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    n_sensors = 50
    latent_dim = 64
    n_states = 3
    
    sensor_x = torch.linspace(0.0, 1.0, n_sensors).to(device)
    model = utils.DeepONet(n_sensors=n_sensors, n_states=n_states, latent_dim=latent_dim).to(device)
    
    try:
        model.load_state_dict(torch.load("deeponet_quantum.pth", map_location=device, weights_only=True))
        print("Model loaded successfully from 'deeponet_quantum.pth'")
    except FileNotFoundError:
        print("Error: Model not found.")
        return

    model.eval()

    test_coefficients = torch.randn(1, 5, device=device)
    V_test_sensors = utils.evaluate_source(test_coefficients, sensor_x)
    
    test_x = torch.linspace(0.0, 1.0, 500, device=device).view(1, 500, 1)
    test_x.requires_grad_(True)
    
    V_test_grid = utils.evaluate_source(test_coefficients, test_x)
    
    # Evaluate the model on the test data
    raw_out = model(V_test_sensors, test_x)
    test_x_sq = test_x.squeeze(-1) 
    
    x_numpy = test_x_sq[0].detach().cpu().numpy()
    V_numpy = V_test_grid.squeeze(-1)[0].detach().cpu().numpy()
    
    # Solve the Schrödinger equation using FDM for comparison
    E_exact_all, psi_exact_all = utils.solve_schrodinger_fdm(x_numpy, V_numpy, n_states=n_states)
    
    psi_preds, E_preds = [], []
    
# Przygotuj puste listy na surowe predykcje z modelu
    psi_preds_unsorted = []
    E_preds_unsorted = []
    
    # 1. FAZA EWALUACJI I OBLICZANIA ENERGII DLA KAŻDEGO STANU 
    # (Tutaj nie przejmujemy się jeszcze kolejnością z jaką sieć wyrzuciła stany)
    for i in range(n_states):
        raw_i = raw_out[..., i] # [1, 500]
        
        # Ansatz
        psi_pred = test_x_sq * (1.0 - test_x_sq) * raw_i
        
        # Normalizacja
        norm_factor = torch.trapezoid(psi_pred**2, x=test_x_sq, dim=1) 
        psi_n = psi_pred / torch.sqrt(norm_factor).unsqueeze(1)
        
        # Pochodne do energii
        psi_x = torch.autograd.grad(psi_n, test_x, grad_outputs=torch.ones_like(psi_n), create_graph=True)[0]
        psi_xx = torch.autograd.grad(psi_x, test_x, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0].squeeze(-1)
        
        H_psi = -0.5 * psi_xx + (V_test_grid.squeeze(-1) * psi_n)
        E_pred = torch.trapezoid(psi_n * H_psi, x=test_x_sq, dim=1)
        
        psi_preds_unsorted.append(psi_n[0])
        E_preds_unsorted.append(E_pred[0])

    # 2. FAZA SORTOWANIA WYNIKÓW WEDŁUG ENERGII
    # torch.argsort tworzy tensor z indeksami, które ułożyłyby energie od najmniejszej do największej.
    # Używamy tej samej listy indeksów do uporządkowania zarówno energii jak i samych funkcji falowych.
    E_tensor = torch.stack(E_preds_unsorted)
    sort_indices = torch.argsort(E_tensor)
    
    psi_preds = [psi_preds_unsorted[idx] for idx in sort_indices]
    E_preds = [E_preds_unsorted[idx] for idx in sort_indices]

    # 3. FAZA WERYFIKACJI Z GROUND TRUTH (FDM) I OBLICZANIA BŁĘDÓW
    # Teraz mamy pewność, że predykcje ułożone są w tej samej kolejności (0, 1, 2) co z klasycznego solvera.
    for i in range(n_states):
        psi_pred_np = psi_preds[i].detach().cpu().numpy()
        psi_exact = psi_exact_all[i]
        
        E_exact = E_exact_all[i]
        E_pred_val = E_preds[i].item()
        
        # Dopasowanie fazy przed policzeniem błędu (aby sinus nie został porównany z minusem sinusem)
        if np.sum(psi_pred_np * psi_exact) < 0:
            psi_pred_np = -psi_pred_np 
            
        error_l2 = np.linalg.norm(psi_pred_np - psi_exact) / np.linalg.norm(psi_exact)
        error_E = abs(E_pred_val - E_exact) / abs(E_exact)
        
        print(f"--- STATE n={i} ---")
        print(f"Relative L2 error for psi(x): {error_l2:.4e}")
        print(f"Relative energy error E:      {error_E:.4e}\n")
    
    # 4. PLOTTOWANIE 
    # Funkcja przyjmuje teraz odpowiednio połączone w pary i posortowane predykcje.
    utils.plot_quantum_comparison(
        x=test_x_sq[0], 
        V=V_test_grid.squeeze(-1)[0], 
        psi_preds=psi_preds,
        E_preds=E_preds,
        psi_exacts=psi_exact_all,
        E_exacts=E_exact_all,
        n_states=n_states
    )

if __name__ == "__main__":
    main()