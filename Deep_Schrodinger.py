import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import utils

n_sensors = 50
n_collocation = 100  # number of collocation points for physics-informed loss
sensor_x = torch.linspace(0.0, 1.0, n_sensors)  # sensor points = positions of f-values (f representation) [M]

# ---------------------------------------------------------
# Training DeepONet
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = utils.DeepONet(
    n_sensors=n_sensors,
    latent_dim=64,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_history = []
steps_per_epoch = 50
for epoch in range(500):
    model.train()
    epoch_loss = 0.0

    # We generate fresh functions and fresh collocation points every step.
    for _ in range(steps_per_epoch):
        _, f_sensors, x_c, f_c = utils.sample_problem_batch(
            batch_size=32,
            sensor_x=sensor_x,
            n_collocation=n_collocation,
            n_modes=5,
            device=device,
        )

        optimizer.zero_grad(set_to_none=True)
        loss_pde = utils.physics_loss(model, f_sensors, x_c, f_c)
        loss_bc = utils.boundary_loss(model, f_sensors)  # here respected by definition of physics_loss, but we keep it separate for clarity
        loss = loss_pde + loss_bc
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % 10 == 0:
        average_loss = epoch_loss / steps_per_epoch
        loss_history.append(average_loss)
        print(f"epoch={epoch:4d}, physics_loss={average_loss:.4e}")


# ---------------------------------------------------------
# Test using new function
# ---------------------------------------------------------

model.eval()

f_test, query_x, u_exact = utils.generate_functions(
    n_functions=1,
    sensor_x=sensor_x,
    query_x=torch.linspace(0.0, 1.0, 100),
    n_modes=5,
)
with torch.no_grad():
    u_pred = model(
        f_test.to(device),
        query_x.to(device),
    ).cpu()
relative_error = (
    torch.linalg.norm(u_pred - u_exact) / torch.linalg.norm(u_exact)
)
print("Relative L2 error:", relative_error.item())
utils.plot_loss(loss_history, no_epochs_per_tick=10)
utils.plot_predictions(query_x, u_exact, u_pred, [sensor_x, f_test])