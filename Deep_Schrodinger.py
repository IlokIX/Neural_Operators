import torch
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_sensors = 50
n_collocation = 2000
n_states = 3  
sensor_x = torch.linspace(0.0, 1.0, n_sensors).to(device)

model = utils.DeepONet(n_sensors=n_sensors, n_states=n_states, latent_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


#---------------------------------------------------------
# Set hyperparameters for training
epochs = 200
steps_per_epoch = 50
w_pde = 1.0
w_norm = 10.0
w_ortho = 10.0
#---------------------------------------------------------

history_pde, history_norm, history_ortho, history_total = [], [], [], []



for epoch in range(epochs):
    model.train()
    epoch_pde, epoch_norm, epoch_ortho, epoch_total = 0.0, 0.0, 0.0, 0.0

    for _ in range(steps_per_epoch):
        # Generate random coefficients for the potential and evaluate the source term at sensor locations
        coefficients = torch.randn(32, 5, device=device)
        V_sensors = utils.evaluate_source(coefficients, sensor_x) 
        
        x_c = torch.rand(32, n_collocation, 1, device=device)
        V_c = utils.evaluate_source(coefficients, x_c)

        optimizer.zero_grad(set_to_none=True)
        
        # Call the loss function from the utils file
        loss_pde, loss_norm, loss_ortho, E = utils.loss_functions(
            model, V_sensors, x_c, V_c,
        )
        
        loss = (w_pde * loss_pde) + (w_norm * loss_norm) + (w_ortho * loss_ortho)
        loss.backward()
        optimizer.step()
        
        epoch_pde += loss_pde.item()
        epoch_norm += loss_norm.item()
        epoch_ortho += loss_ortho.item()
        epoch_total += loss.item()

    if epoch % 10 == 0:
        avg_total = epoch_total / steps_per_epoch
        history_total.append(avg_total)
        history_pde.append(epoch_pde / steps_per_epoch)
        history_norm.append(epoch_norm / steps_per_epoch)
        history_ortho.append(epoch_ortho / steps_per_epoch)
        
        print(f"epoch={epoch:4d} | Total: {avg_total:.3e} | PDE: {history_pde[-1]:.3e} | Norm: {history_norm[-1]:.3e} | Ortho: {history_ortho[-1]:.3e}")


# Save the trained model
torch.save(model.state_dict(), "deeponet_quantum.pth")
print("Saved model weights to file 'deeponet_quantum.pth'")

# Plot the loss history
utils.plot_separated_losses(
    history_total, 
    history_pde, 
    history_norm, 
    history_ortho, 
    no_epochs_per_tick=10
)