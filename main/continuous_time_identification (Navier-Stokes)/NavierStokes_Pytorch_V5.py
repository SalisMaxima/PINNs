"""
This code implements a Physics-Informed Neural Network (PINN) designed to learn the non-linear parameters 
(lambda_1 and lambda_2) from a given dataset and predict the fluid flow of the system when provided with 
spatial and temporal coordinates (x, y, t).

The PINN integrates the physical laws governing fluid dynamics (Navier-Stokes equations) directly into the 
training process. By doing so, it leverages both the data and the underlying physical principles to achieve 
more accurate and physically consistent predictions.

The model architecture is dynamically constructed based on the specified layers and activation function. 
The training process involves optimizing the network to minimize the difference between the predicted and 
true velocity components, as well as the residuals of the Navier-Stokes equations. Additionally, the model 
learns the parameters lambda_1 (representing the convective term coefficient) and lambda_2 (representing 
the diffusive term coefficient).

This code also includes functionality for adding noise to the training data, tracking the loss during 
training, and saving the trained model at regular intervals. The loss history can be visualized by plotting 
the training loss over epochs.

Example usage of this code involves specifying the network architecture, loading the dataset, and calling 
the train_pinn function to start the training process.

Classes and Functions:
- PhysicsInformedNN: Defines the neural network architecture and forward pass, including the computation 
  of velocity components and Navier-Stokes residuals.
- train_pinn: Function to train the PINN, including data preparation, noise addition, model optimization, 
  and loss tracking.

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import scipy.io
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
        
        # Initialize lambda parameters
        self.lambda_1 = nn.Parameter(torch.tensor([0.0], device=self.device))
        self.lambda_2 = nn.Parameter(torch.tensor([0.0], device=self.device))

    def forward(self, x, y, t):
        # Concatenate inputs and pass through the model
        cat_input = torch.cat([x, y, t], dim=1)
        output = self.model(cat_input)
        
        # Split the output into psi and pressure (p)
        psi = output[:, 0:1]
        p = output[:, 1:2]
        
        # Calculate velocity components u and v
        u = grad(psi.sum(), y, create_graph=True)[0]
        v = -grad(psi.sum(), x, create_graph=True)[0]
        
        # Calculate partial derivatives for the Navier-Stokes equations
        u_t = grad(u.sum(), t, create_graph=True)[0]
        u_x = grad(u.sum(), x, create_graph=True)[0]
        u_y = grad(u.sum(), y, create_graph=True)[0]
        u_xx = grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = grad(u_y.sum(), y, create_graph=True)[0]

        v_t = grad(v.sum(), t, create_graph=True)[0]
        v_x = grad(v.sum(), x, create_graph=True)[0]
        v_y = grad(v.sum(), y, create_graph=True)[0]
        v_xx = grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = grad(v_y.sum(), y, create_graph=True)[0]
        
        p_x = grad(p.sum(), x, create_graph=True)[0]
        p_y = grad(p.sum(), y, create_graph=True)[0]

        # Calculate the residuals of the Navier-Stokes equations
        f_u = u_t + self.lambda_1 * (u * u_x + v * u_y) + p_x - self.lambda_2 * (u_xx + u_yy)
        f_v = v_t + self.lambda_1 * (u * v_x + v * v_y) + p_y - self.lambda_2 * (v_xx + v_yy)
        
        return u, v, p , f_u, f_v

    def loss_function(self, u_true, v_true, outputs):
        # Unpack the outputs
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = outputs
        # Calculate mean squared errors for velocity and residuals
        mse_u = torch.mean((u_true - u_pred) ** 2)
        mse_v = torch.mean((v_true - v_pred) ** 2)
        mse_f_u = torch.mean(f_u_pred ** 2)
        mse_f_v = torch.mean(f_v_pred ** 2)
        # Total loss is the sum of these errors
        total_loss = (mse_u + mse_v) + (mse_f_u + mse_f_v)
        return total_loss

def train_pinn(data_path, layers, epochs=200000, batch_size=5000, lr=0.001, noise_level=0.0, save_path="model.pth", track_loss=False):
    # Load the data from the .mat file
    data = scipy.io.loadmat(data_path)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Cuda available! Training on GPU.")
    else:
        device = torch.device("cpu")
        print("Training on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract and process data
    U_star = torch.tensor(data['U_star'], dtype=torch.float32).to(device)
    P_star = torch.tensor(data['p_star'], dtype=torch.float32).to(device)
    t_star = torch.tensor(data['t'], dtype=torch.float32).to(device)
    X_star = torch.tensor(data['X_star'], dtype=torch.float32).to(device)

    N = X_star.shape[0]
    T = t_star.shape[0]
    XX = X_star[:, 0:1].repeat(1, T)
    YY = X_star[:, 1:2].repeat(1, T)
    TT = t_star.repeat(1, N).T

    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = U_star[:, 0, :].flatten()[:, None]
    v = U_star[:, 1, :].flatten()[:, None]

    # Initialize the model and optimizer
    model = PhysicsInformedNN(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Select a random batch of training data
    N_train = batch_size
    idx = np.random.choice(N * T, N_train, replace=False)
    x_train = x[idx, :].requires_grad_(True)
    y_train = y[idx, :].requires_grad_(True)
    t_train = t[idx, :].requires_grad_(True)
    u_train = u[idx, :]
    v_train = v[idx, :]

    # Add noise to the training data if specified
    if noise_level > 0:
        u_train += noise_level * torch.std(u_train) * torch.randn_like(u_train)
        v_train += noise_level * torch.std(v_train) * torch.randn_like(v_train)
        print("Training data with noise added successfully!")

    # Require gradients for the training data
    u_train = u_train.clone().requires_grad_(True)
    v_train = v_train.clone().requires_grad_(True)

    training_data = [(x_train, y_train, t_train, u_train, v_train)]
    start_time = time.time()
    model.train()

    if track_loss:
        loss_history = []

    # Training loop
    for epoch in range(epochs):
        for x, y, t, u, v in training_data:
            optimizer.zero_grad()
            u_pred, v_pred, p_pred, f_u_pred, f_v_pred = model(x, y, t)
            loss = model.loss_function(u, v, (u_pred, v_pred, p_pred, f_u_pred, f_v_pred))
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
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    data_path = '../Data/cylinder_nektar_wake.mat'
    save_path = "model_xavier_loss_curve.pth"
    train_pinn(data_path, layers, epochs=200000, batch_size=5000, lr=0.001, noise_level=0, save_path=save_path, track_loss=True)
