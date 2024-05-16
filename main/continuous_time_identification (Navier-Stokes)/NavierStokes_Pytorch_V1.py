# Imports necessary libraries and modules for handling arrays, machine learning, and data visualization
import numpy as np
import torch
import torch.nn as nn
import scipy.io
import matplotlib.pyplot as plt
from torch.autograd import grad
import time

# Set seeds for reproducibility of experiments
np.random.seed(1234)
torch.manual_seed(1234)

# Define the PhysicsInformedNN class to encapsulate the neural network for solving PDEs
class PhysicsInformedNN(torch.nn.Module):
    def __init__(self, layers, activation='tanh'):
        super(PhysicsInformedNN, self).__init__()
        self.model = nn.Sequential()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        activation_function = nn.Tanh() if activation == 'tanh' else nn.ReLU()

        for i in range(len(layers)-1):
            self.model.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.model.add_module(f"{activation}_{i}", activation_function)
        
        self.model.to(self.device)
        self.lambda_1 = nn.Parameter(torch.tensor([0.0], device=self.device))
        self.lambda_2 = nn.Parameter(torch.tensor([0.0], device=self.device))
    # Define the forward pass of the neural network
    def forward(self, x, y, t):
        # Ensure inputs are on the correct device
        cat_input = torch.cat([x, y, t], dim=1)
        output = self.model(cat_input)
        
        psi = output[:, 0:1] # Extract the first output as the stream function and ensure it has gradient enabled
        p = output[:, 1:2] # Extract the second output as the pressure field and ensure it has gradient enabled
        
        u = grad(psi.sum(), y, create_graph=True)[0] # Compute the u velocity component using automatic differentiation
        v = -grad(psi.sum(), x, create_graph=True)[0] # Compute the v velocity component using automatic differentiation
        
        u_t = grad(u.sum(), t, create_graph=True)[0] # Compute the time derivative of u
        u_x = grad(u.sum(), x, create_graph=True)[0] # Compute the spatial derivative of u with respect to x
        u_y = grad(u.sum(), y, create_graph=True)[0] # Compute the spatial derivative of u with respect to y
        u_xx = grad(u_x.sum(), x, create_graph=True)[0] # Compute the second spatial derivative of u with respect to x
        u_yy = grad(u_y.sum(), y, create_graph=True)[0] # Compute the second spatial derivative of u with respect to y

        v_t = grad(v.sum(), t, create_graph=True)[0] # Compute the time derivative of v
        v_x = grad(v.sum(), x, create_graph=True)[0] # Compute the spatial derivative of v with respect to x
        v_y = grad(v.sum(), y, create_graph=True)[0] # Compute the spatial derivative of v with respect to y
        v_xx = grad(v_x.sum(), x, create_graph=True)[0] # Compute the second spatial derivative of v with respect to x
        v_yy = grad(v_y.sum(), y, create_graph=True)[0] # Compute the second spatial derivative of v with respect to y
        
        p_x = grad(p.sum(), x, create_graph=True)[0] # Compute the spatial derivative of p with respect to x
        p_y = grad(p.sum(), y, create_graph=True)[0] # Compute the spatial derivative of p with respect to y

        f_u = u_t + self.lambda_1 * (u * u_x + v * u_y) + p_x - self.lambda_2 * (u_xx + u_yy) # Compute the residual for the u velocity component
        f_v = v_t + self.lambda_1 * (u * v_x + v * v_y) + p_y - self.lambda_2 * (v_xx + v_yy) # Compute the residual for the v velocity component
        
        return u, v, p , f_u, f_v

    def loss_function(self, u_true, v_true, outputs):
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = outputs
        mse_u = torch.mean((u_true - u_pred) ** 2)
        mse_v = torch.mean((v_true - v_pred) ** 2)
        mse_f_u = torch.mean(f_u_pred ** 2)
        mse_f_v = torch.mean(f_v_pred ** 2)
        total_loss = mse_u + mse_v + mse_f_u + mse_f_v
        return total_loss


# Main execution logic to setup the model and start training
if __name__ == "__main__":
    if torch.cuda.is_available():
        print("CUDA is available! Testing with a simple tensor operation.")
        # wait for 5 seconds
        time.sleep(5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    data = scipy.io.loadmat('../Data/cylinder_nektar_wake.mat')
  
    # Extract and process data
    U_star = torch.tensor(data['U_star'], dtype=torch.float32).to(device)
    # print the size of U_star in bytes
    #print(U_star.element_size() * U_star.nelement())
    # print the size of U_star in GB
    #print(U_star.element_size() * U_star.nelement() / 1024**3)
    P_star = torch.tensor(data['p_star'], dtype=torch.float32).to(device)
    # print the size of p_star in bytes
    #print(P_star.element_size() * P_star.nelement())
    # print the size of p_star in GB
    #print(P_star.element_size() * P_star.nelement() / 1024**3)
    t_star = torch.tensor(data['t'], dtype=torch.float32).to(device)
    # print the size of t_star in bytes
    #print(t_star.element_size() * t_star.nelement())
    # print the size of t_star in GB
    #print(t_star.element_size() * t_star.nelement() / 1024**3)
    X_star = torch.tensor(data['X_star'], dtype=torch.float32).to(device)
    # print the size of X_star in bytes
    #print(X_star.element_size() * X_star.nelement())
    # print the size of X_star in GB
    #print(X_star.element_size() * X_star.nelement() / 1024**3)
    
    # Flatten and prepare data
    N = X_star.shape[0]
    T = t_star.shape[0]
    XX = X_star[:, 0:1].repeat(1, T) # dimensions are N x T
    YY = X_star[:, 1:2].repeat(1, T) # dimensions are N x T
    TT = t_star.repeat(1,N).T # dimensions are N x T

    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = U_star[:, 0, :].flatten()[:, None]
    v = U_star[:, 1, :].flatten()[:, None]
    
    model = PhysicsInformedNN(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    N_train = 5000
    idx = np.random.choice(N * T, N_train, replace=False)
    x_train = x[idx, :].requires_grad_(True)
    y_train = y[idx, :].requires_grad_(True)
    t_train = t[idx, :].requires_grad_(True)
    u_train = u[idx, :]
    v_train = v[idx, :]

    # Add noise to the training data u_train and v_train
    #noise_level = 0.05
    #u_train += noise_level * torch.std(u_train) * torch.randn_like(u_train)
    #v_train += noise_level * torch.std(v_train) * torch.randn_like(v_train)
    #print("Training data with noise added successfully!")
    # Require gradients for the training data
    u_train = u_train.clone().requires_grad_(True)
    v_train = v_train.clone().requires_grad_(True)
    
    ############################# Training ######################################
    training_data = [(x_train, y_train, t_train, u_train, v_train)]
    start_time = time.time()
    model.train()
    for epoch in range(200000):
        for x, y, t, u, v in training_data:
            optimizer.zero_grad()
            u_pred, v_pred, p_pred, f_u_pred, f_v_pred = model(x, y, t)
            loss = model.loss_function(u, v, (u_pred, v_pred, p_pred, f_u_pred, f_v_pred))
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")
    
    # Save the model
    torch.save(model.state_dict(), "model2.pth")
    print("Model saved successfully!")
