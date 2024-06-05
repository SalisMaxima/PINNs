import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import scipy.io
import matplotlib.pyplot as plt
from torch.autograd import grad
import time
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
import plotly.io as pio

# Set seeds for reproducibility
np.random.seed(1234)
torch.manual_seed(1234)

class PhysicsInformedNN(nn.Module):
    def __init__(self, layers, rho1=1.0, rho2=1.0, activation='tanh', dropout_prob=0.0):
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
                # Add linear, activation, and dropout layers to the model
                self.model.add_module(f"linear_{i}", linear_layer)
                self.model.add_module(f"{activation}_{i}", activation_function)
                if dropout_prob > 0.0:
                    self.model.add_module(f"dropout_{i}", nn.Dropout(dropout_prob))
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

        # Store the rho parameters
        self.rho1 = rho1
        self.rho2 = rho2

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
        # Total loss is the sum of these errors weighted by rho1 and rho2
        total_loss = self.rho1 * (mse_u + mse_v) + self.rho2 * (mse_f_u + mse_f_v)
        return total_loss, mse_u, mse_v, mse_f_u, mse_f_v

def plot_interactive_loss_curves(train_loss_history, val_loss_history, mse_u_history, mse_v_history, mse_f_u_history, mse_f_v_history, epochs, save_path):
    # Plot training and validation loss curves
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=list(range(epochs)), y=train_loss_history, mode='lines', name='Training Loss', line=dict(color='blue', width=0.5)))
    fig1.add_trace(go.Scatter(x=list(range(epochs)), y=val_loss_history, mode='lines', name='Validation Loss', line=dict(color='orange', width=0.5, dash='dash')))
    fig1.update_layout(title='Training and Validation Loss Over Epochs', xaxis_title='Epochs', yaxis_title='Loss')
    loss_curve_name = f"Loss_curve_{save_path.split('.')[0]}.html"
    pio.write_html(fig1, file=loss_curve_name, auto_open=True)
    print(f"Interactive loss curve saved successfully as {loss_curve_name}!")

    # Plot individual MSE loss curves
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=list(range(epochs)), y=mse_u_history, mode='lines+markers', name='MSE U', line=dict(color='red', width=0.2), marker=dict(size=2)))
    fig2.add_trace(go.Scatter(x=list(range(epochs)), y=mse_v_history, mode='lines+markers', name='MSE V', line=dict(color='green', width=0.2, dash='dash'), marker=dict(size=2, symbol='square')))
    fig2.add_trace(go.Scatter(x=list(range(epochs)), y=mse_f_u_history, mode='lines+markers', name='MSE f_u', line=dict(color='blue', width=0.2, dash='dot'), marker=dict(size=2, symbol='triangle-up')))
    fig2.add_trace(go.Scatter(x=list(range(epochs)), y=mse_f_v_history, mode='lines+markers', name='MSE f_v', line=dict(color='purple', width=0.2, dash='dot'), marker=dict(size=2, symbol='x')))
    fig2.update_layout(title='Individual MSE Losses Over Epochs', xaxis_title='Epochs', yaxis_title='MSE Loss')
    mse_loss_curve_name = f"MSE_Loss_curve_{save_path.split('.')[0]}.html"
    pio.write_html(fig2, file=mse_loss_curve_name, auto_open=True)
    print(f"Interactive individual MSE loss curves saved successfully as {mse_loss_curve_name}!")

def train_pinn(data_path, layers, rho1=1.0, rho2=1.0, epochs=200000, batch_size=5000, lr=0.001, noise_level=0.0, save_path="model.pth", track_loss=False, dropout_prob=0.0):
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

    # Select a random batch of training data
    N_train = batch_size
    idx = np.random.choice(N * T, N_train, replace=False)
    x_train, x_val = train_test_split(x[idx, :], test_size=0.2, random_state=1234)
    y_train, y_val = train_test_split(y[idx, :], test_size=0.2, random_state=1234)
    t_train, t_val = train_test_split(t[idx, :], test_size=0.2, random_state=1234)
    u_train, u_val = train_test_split(u[idx, :], test_size=0.2, random_state=1234)
    v_train, v_val = train_test_split(v[idx, :], test_size=0.2, random_state=1234)

    # Initialize the model and optimizer
    model = PhysicsInformedNN(layers, rho1=rho1, rho2=rho2, dropout_prob=dropout_prob).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # Add noise to the training data if specified
    if noise_level > 0:
        u_train += noise_level * torch.std(u_train) * torch.randn_like(u_train)
        v_train += noise_level * torch.std(v_train) * torch.randn_like(v_train)
        print("Training data with noise added successfully!")

    # Require gradients for the training and validation data
    x_train.requires_grad_(True)
    y_train.requires_grad_(True)
    t_train.requires_grad_(True)
    u_train.requires_grad_(True)
    v_train.requires_grad_(True)

    x_val.requires_grad_(True)
    y_val.requires_grad_(True)
    t_val.requires_grad_(True)
    u_val.requires_grad_(True)
    v_val.requires_grad_(True)

    start_time = time.time() # Track time for training
    model.train()

    if track_loss:
        train_loss_history = []
        val_loss_history = []
        mse_u_history = []
        mse_v_history = []
        mse_f_u_history = []
        mse_f_v_history = []

    # Training loop
    for epoch in range(epochs):
        # Training step
        optimizer.zero_grad()
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = model(x_train, y_train, t_train)
        loss, mse_u, mse_v, mse_f_u, mse_f_v = model.loss_function(u_train, v_train, (u_pred, v_pred, p_pred, f_u_pred, f_v_pred))
        loss.backward()
        optimizer.step()

        if track_loss:
            # Validation step
            model.eval()
            u_val_pred, v_val_pred, p_val_pred, f_u_val_pred, f_v_val_pred = model(x_val, y_val, t_val)
            val_loss, val_mse_u, val_mse_v, val_mse_f_u, val_mse_f_v = model.loss_function(u_val, v_val, (u_val_pred, v_val_pred, p_val_pred, f_u_val_pred, f_v_val_pred))
            model.train()
            train_loss_history.append(loss.item())
            val_loss_history.append(val_loss.item())
            mse_u_history.append(mse_u.item())
            mse_v_history.append(mse_v.item())
            mse_f_u_history.append(mse_f_u.item())
            mse_f_v_history.append(mse_f_v.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train Loss = {loss.item()}, Val Loss = {val_loss.item()}")
        if epoch % 1000 == 0:  # Save the model every 1000 epochs
            torch.save(model.state_dict(), save_path)
            print(f"Model saved at epoch {epoch}")
    
    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")
    torch.save(model.state_dict(), save_path)
    print("Model saved successfully!")

    # Plot and save the loss curve if tracking loss
    if track_loss:
        plot_interactive_loss_curves(train_loss_history, val_loss_history, mse_u_history, mse_v_history, mse_f_u_history, mse_f_v_history, epochs, save_path)
        # Plot and save the training and validation loss curves
        lw = 0.1
        msz = 0.01
        plt.figure(figsize=(10, 5))
        plt.semilogy(range(epochs), train_loss_history, label="Training Loss", linestyle='-', color='blue', linewidth=lw+0.2, markersize=msz)
        plt.semilogy(range(epochs), val_loss_history, label="Validation Loss", linestyle='--', color='orange', linewidth=lw + 0.2, markersize=msz)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        loss_curve_name = f"Loss_curve_{save_path.split('.')[0]}.png"
        plt.savefig(loss_curve_name)
        plt.show()
        print(f"Loss curve saved successfully as {loss_curve_name}!")

        # Plot and save the individual loss curves
        plt.figure(figsize=(10, 5))
        
        plt.semilogy(range(epochs), mse_u_history, label="MSE U", linestyle='-', marker='o', color='red', linewidth=lw, markersize=msz)
        plt.semilogy(range(epochs), mse_v_history, label="MSE V", linestyle='--', marker='s', color='green', linewidth=lw, markersize=msz)
        plt.semilogy(range(epochs), mse_f_u_history, label="MSE f_u", linestyle='-.', marker='^', color='blue', linewidth=lw, markersize=msz)
        plt.semilogy(range(epochs), mse_f_v_history, label="MSE f_v", linestyle=':', marker='x', color='purple', linewidth=lw, markersize=msz)
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")
        plt.title("Individual MSE Losses Over Epochs")
        plt.legend()
        plt.grid(True)
        mse_loss_curve_name = f"MSE_Loss_curve_{save_path.split('.')[0]}.png"
        plt.savefig(mse_loss_curve_name)
        plt.show()
        print(f"Individual MSE loss curves saved successfully as {mse_loss_curve_name}!")
    return model, train_loss_history, val_loss_history, mse_u_history, mse_v_history, mse_f_u_history, mse_f_v_history
# Example usage
if __name__ == "__main__":
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    data_path = '../Data/cylinder_nektar_wake.mat'
    save_path = "model_with_dropout.pth"
    rho1 = 1.0  # Set your rho1 value here
    rho2 = 1.0  # Set your rho2 value here
    dropout_prob = 0  # Set your dropout probability here
    model, train_loss_history, val_loss_history, mse_u_history, mse_v_history, mse_f_u_history, mse_f_v_history = train_pinn(data_path, layers, rho1=rho1, rho2=rho2, epochs=200000, batch_size=5000, lr=0.001, noise_level=0, save_path=save_path, track_loss=True, dropout_prob=dropout_prob)
