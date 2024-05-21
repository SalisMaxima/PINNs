import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt
from torch.autograd import grad
import time

# Set seeds for reproducibility
np.random.seed(1234)
torch.manual_seed(1234)

class PhysicsInformedNN(nn.Module):
    def __init__(self, layers, activation='tanh'):
        super(PhysicsInformedNN, self).__init__()
        self.model = nn.Sequential()
        
        # Determine the device to run on (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Choose the activation function based on the input parameter
        activation_function = nn.Tanh() if activation == 'tanh' else nn.ReLU()

        # Build the neural network
        for i in range(len(layers) - 1):
            linear_layer = nn.Linear(layers[i], layers[i + 1])
            if i < len(layers) - 2:
                # Add linear and activation layers to the model
                self.model.add_module(f"linear_{i}", linear_layer)
                self.model.add_module(f"{activation}_{i}", activation_function)
            else:
                # Add the final linear layer
                self.model.add_module(f"linear_{i}", linear_layer)
            # Apply Xavier initialization to the linear layer
            init.xavier_uniform_(linear_layer.weight)
            if linear_layer.bias is not None:
                init.constant_(linear_layer.bias, 0.0)
        
        # Move the model to the appropriate device
        self.model.to(self.device)
        
        # Initialize Lorenz system parameters
        self.sigma = nn.Parameter(torch.tensor([10.0], device=self.device))
        self.rho = nn.Parameter(torch.tensor([28.0], device=self.device))
        self.beta = nn.Parameter(torch.tensor([8.0/3.0], device=self.device))

    def forward(self, t):
        # Pass time through the model
        output = self.model(t)
        
        # Split the output into x, y, z
        x = output[:, 0:1]
        y = output[:, 1:2]
        z = output[:, 2:3]
        
        return x, y, z

    def loss_function(self, t, true_values):
        # Unpack true values
        x_true, y_true, z_true = true_values[:, 0:1], true_values[:, 1:2], true_values[:, 2:3]
        
        # Predicted values
        x_pred, y_pred, z_pred = self(t)

        # Data loss
        mse_x = torch.mean((x_true - x_pred) ** 2)
        mse_y = torch.mean((y_true - y_pred) ** 2)
        mse_z = torch.mean((z_true - z_pred) ** 2)
        
        # Calculate derivatives
        x_t = grad(x_pred, t, grad_outputs=torch.ones_like(x_pred), create_graph=True)[0]
        y_t = grad(y_pred, t, grad_outputs=torch.ones_like(y_pred), create_graph=True)[0]
        z_t = grad(z_pred, t, grad_outputs=torch.ones_like(z_pred), create_graph=True)[0]
        
        # Physics loss
        f1 = x_t - self.sigma * (y_pred - x_pred)
        f2 = y_t - x_pred * (self.rho - z_pred) + y_pred
        f3 = z_t - (x_pred * y_pred - self.beta * z_pred)
        
        mse_f1 = torch.mean(f1 ** 2)
        mse_f2 = torch.mean(f2 ** 2)
        mse_f3 = torch.mean(f3 ** 2)
        
        # Total loss
        total_loss = mse_x + mse_y + mse_z + mse_f1 + mse_f2 + mse_f3
        return total_loss

def train_pinn(t_data, true_values, layers, epochs=10000, lr=0.001, save_path="model.pth", track_loss=False):
    # Determine the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Convert data to tensors
    t_data = torch.tensor(t_data, dtype=torch.float32).to(device).unsqueeze(1)
    true_values = torch.tensor(true_values, dtype=torch.float32).to(device)
    
    # Ensure t_data requires gradients
    t_data.requires_grad = True
    
    # Initialize the model and optimizer
    model = PhysicsInformedNN(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if track_loss:
        loss_history = []
    
    start_time = time.time()
    model.train()

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = model.loss_function(t_data, true_values)
        loss.backward()
        optimizer.step()
        
        if track_loss:
            loss_history.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")
        if epoch % 1000 == 0:  # Save the model every 1000 epochs
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch}")
    
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")
    torch.save(model.state_dict(), save_path)
    print("Model saved successfully!")

    # Plot and save the loss curve if tracking loss
    if track_loss:
        loss_curve_name = f"Loss_curve_{save_path.split('.')[0]}.png"
        plt.figure(figsize=(10, 5))
        plt.plot(range(epochs), loss_history, label="Training Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.savefig(loss_curve_name)
        plt.show()
        print(f"Loss curve saved successfully as {loss_curve_name}!")

# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    def lorenz(t, state, sigma, rho, beta):
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]
    
    # Parameters
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    
    # Initial condition
    initial_state = [1.0, 1.0, 1.0]
    
    # Time points where solution is computed
    t_span = (0, 25)
    t_eval = np.linspace(t_span[0], t_span[1], 10000)
    
    # Solve the Lorenz system
    from scipy.integrate import solve_ivp
    solution = solve_ivp(lorenz, t_span, initial_state, args=(sigma, rho, beta), t_eval=t_eval)
    
    # Prepare the data
    t_data = solution.t
    true_values = np.vstack((solution.y[0], solution.y[1], solution.y[2])).T
    
    layers = [1, 64, 64, 64, 3]
    save_path = "lorenz_model.pth"
    train_pinn(t_data, true_values, layers, epochs=10000, lr=0.01, save_path=save_path, track_loss=True)
