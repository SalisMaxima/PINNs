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
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        activation_function = nn.Tanh() if activation == 'tanh' else nn.ReLU()

        for i in range(len(layers) - 1):
            linear_layer = nn.Linear(layers[i], layers[i + 1])
            if i < len(layers) - 2:
                self.model.add_module(f"linear_{i}", linear_layer)
                self.model.add_module(f"{activation}_{i}", activation_function)
            else:
                self.model.add_module(f"linear_{i}", linear_layer)
            # Apply Xavier initialization to the linear layer
            init.xavier_uniform_(linear_layer.weight)
            if linear_layer.bias is not None:
                init.constant_(linear_layer.bias, 0.0)
        
        self.model.to(self.device)
        self.lambda_1 = nn.Parameter(torch.tensor([0.0], device=self.device))
        self.lambda_2 = nn.Parameter(torch.tensor([0.0], device=self.device))

    def forward(self, x, y, t):
        cat_input = torch.cat([x, y, t], dim=1)
        output = self.model(cat_input)
        
        psi = output[:, 0:1]
        p = output[:, 1:2]
        
        u = grad(psi.sum(), y, create_graph=True)[0]
        v = -grad(psi.sum(), x, create_graph=True)[0]
        
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

        f_u = u_t + self.lambda_1 * (u * u_x + v * u_y) + p_x - self.lambda_2 * (u_xx + u_yy)
        f_v = v_t + self.lambda_1 * (u * v_x + v * v_y) + p_y - self.lambda_2 * (v_xx + v_yy)
        
        return u, v, p , f_u, f_v

    def loss_function(self, u_true, v_true, outputs):
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = outputs
        mse_u = torch.mean((u_true - u_pred) ** 2)
        mse_v = torch.mean((v_true - v_pred) ** 2)
        mse_f_u = torch.mean(f_u_pred ** 2)
        mse_f_v = torch.mean(f_v_pred ** 2)
        total_loss = (mse_u + mse_v) + (mse_f_u + mse_f_v)
        return total_loss

def train_pinn(data_path, layers, epochs=200000, batch_size=5000, lr=0.001, noise_level=0.0, save_path="model.pth"):
    data = scipy.io.loadmat(data_path)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Cuda available! Training on GPU.")
    else:
        device = torch.device("cpu")
        print("Training on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    model = PhysicsInformedNN(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    N_train = batch_size
    idx = np.random.choice(N * T, N_train, replace=False)
    x_train = x[idx, :].requires_grad_(True)
    y_train = y[idx, :].requires_grad_(True)
    t_train = t[idx, :].requires_grad_(True)
    u_train = u[idx, :]
    v_train = v[idx, :]

    # Add noise to the training data u_train and v_train
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
    for epoch in range(epochs):
        for x, y, t, u, v in training_data:
            optimizer.zero_grad()
            u_pred, v_pred, p_pred, f_u_pred, f_v_pred = model(x, y, t)
            loss = model.loss_function(u, v, (u_pred, v_pred, p_pred, f_u_pred, f_v_pred))
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")
        if epoch % 1000 == 0:  # Save the model every 1000 epochs
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch}")
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")
    torch.save(model.state_dict(), save_path)
    print("Model saved successfully!")

# Example usage
if __name__ == "__main__":
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    data_path = '../Data/cylinder_nektar_wake.mat'
    save_path = "model_noisy_xavier_01.pth"
    train_pinn(data_path, layers, epochs=200000, batch_size=5000, lr=0.001, noise_level=0.01, save_path=save_path)
