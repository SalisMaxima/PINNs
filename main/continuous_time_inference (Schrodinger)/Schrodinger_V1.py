import numpy as np
import torch
import torch.nn as nn
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

        for i in range(len(layers)-1):
            self.model.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.model.add_module(f"{activation}_{i}", activation_function)
        
        self.model.to(self.device)

    def forward(self, x, t):
        cat_input = torch.cat([x, t], dim=1)
        output = self.model(cat_input)
        u = output[:, 0:1]
        v = output[:, 1:2]
        return u, v

    def loss_function(self, u_true, v_true, u_pred, v_pred, f_u_pred, f_v_pred):
        mse_u = torch.mean((u_true - u_pred) ** 2)
        mse_v = torch.mean((v_true - v_pred) ** 2)
        mse_f_u = torch.mean(f_u_pred ** 2)
        mse_f_v = torch.mean(f_v_pred ** 2)
        total_loss = mse_u + mse_v + mse_f_u + mse_f_v
        return total_loss

    def net_uv(self, x, t):
        u, v = self.forward(x, t)
        u_x = grad(u.sum(), x, create_graph=True)[0]
        v_x = grad(v.sum(), x, create_graph=True)[0]
        u_t = grad(u.sum(), t, create_graph=True)[0]
        v_t = grad(v.sum(), t, create_graph=True)[0]
        u_xx = grad(u_x.sum(), x, create_graph=True)[0]
        v_xx = grad(v_x.sum(), x, create_graph=True)[0]
        return u, v, u_x, v_x, u_t, v_t, u_xx, v_xx

    def net_f_uv(self, x, t):
        u, v, u_x, v_x, u_t, v_t, u_xx, v_xx = self.net_uv(x, t)
        f_u = u_t + 0.5 * v_xx + (u**2 + v**2) * v
        f_v = v_t - 0.5 * u_xx - (u**2 + v**2) * u
        return f_u, f_v

def train_pinn(data_path, layers, epochs=200000, batch_size=5000, lr=0.001, save_path="model.pth"):
    data = scipy.io.loadmat(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    t_star = torch.tensor(data['tt'], dtype=torch.float32).to(device)
    x_star = torch.tensor(data['x'], dtype=torch.float32).to(device)
    Exact = torch.tensor(data['uu'], dtype=torch.complex64).to(device)
    
    u_star = Exact.real
    v_star = Exact.imag

    X, T = torch.meshgrid(x_star, t_star)
    X_star = torch.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    
    u_star = u_star.T.flatten()[:, None]
    v_star = v_star.T.flatten()[:, None]
    
    # Initialize model
    model = PhysicsInformedNN(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training data
    idx = np.random.choice(X_star.shape[0], batch_size, replace=False)
    x_train = X_star[idx, 0:1].requires_grad_(True)
    t_train = X_star[idx, 1:2].requires_grad_(True)
    u_train = u_star[idx, :]
    v_train = v_star[idx, :]
    
    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        u_pred, v_pred = model(x_train, t_train)
        f_u_pred, f_v_pred = model.net_f_uv(x_train, t_train)
        loss = model.loss_function(u_train, v_train, u_pred, v_pred, f_u_pred, f_v_pred)
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
    layers = [2, 100, 100, 100, 100, 2]
    data_path = '../Data/NLS.mat'
    save_path = "model_schrodinger.pth"
    train_pinn(data_path, layers, epochs=50000, batch_size=5000, lr=0.001, save_path=save_path)
