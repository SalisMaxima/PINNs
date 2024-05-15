# 1. Setup and Initialization
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from torch.optim import LBFGS, Adam
from torch.autograd import grad

# Ensure reproducibility
torch.manual_seed(1234)
np.random.seed(1234)

# 2. Physics-Informed Neural Network Setup
class PhysicsInformedNN(torch.nn.Module): # First I will define the neural network class
    def __init__(self, layers, device): # The constructor will take a list of layer sizes and a device
        super(PhysicsInformedNN, self).__init__()
        self.layers = layers
        self.device = device
        self.build_network()

    def build_network(self):
        """Construct the neural network using a list of layers."""
        modules = []
        for i in range(len(self.layers) - 1):
            modules.append(torch.nn.Linear(self.layers[i], self.layers[i+1]))
            if i < len(self.layers) - 2:
                modules.append(torch.nn.Tanh())  # Non-linear activation function
        self.model = torch.nn.Sequential(*modules).to(self.device)

    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)

    def predict(self, x):
        """Predict the outputs given input tensor x."""
        self.model.eval()  # Evaluation mode
        with torch.no_grad():
            return self.forward(x)

    def compute_loss(self, x, y, t, u, v):
        """Compute the loss function combining PINN outputs and MSE for given inputs."""
        # Predict outputs
        u_pred, v_pred = self.forward(torch.cat([x, y, t], dim=1))
        
        # Compute mean squared error on predictions
        mse_loss = torch.mean((u_pred - u) ** 2) + torch.mean((v_pred - v) ** 2)
        
        # Compute residuals of the Navier-Stokes equations (placeholder)
        # This should include the derivation of residuals using autograd and physics constraints
        navier_stokes_residual = torch.tensor([0.0])  # Placeholder
        
        return mse_loss + navier_stokes_residual

# 3. Training Mechanism
def train(model, training_data, epochs, optimizer):
    model.train()
    for epoch in range(epochs):
        for x, y, t, u, v in training_data:
            optimizer.zero_grad()
            loss = model.compute_loss(x, y, t, u, v)
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item()}')

# 4. Data Handling and Main Execution
if __name__ == "__main__":
    N_train = 5000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]  # Example architecture
    model = PhysicsInformedNN(layers, device)
    optimizer = Adam(model.parameters(), lr=0.001)

    # Load and prepare data
    data = scipy.io.loadmat('../Data/cylinder_nektar_wake.mat')
    U_star = data['U_star']  # N x 2 x T
    P_star = data['p_star']  # N x T
    t_star = data['t']       # T x 1
    X_star = data['X_star']  # N x 2
    N = X_star.shape[0]
    T = t_star.shape[0]
    XX = np.tile(X_star[:,0:1], (1,T))  # N x T
    YY = np.tile(X_star[:,1:2], (1,T))  # N x T
    TT = np.tile(t_star, (1,N)).T       # N x T
    UU = U_star[:,0,:]  # N x T
    VV = U_star[:,1,:]  # N x T
    PP = P_star         # N x T
    x = XX.flatten()[:,None]  # NT x 1
    y = YY.flatten()[:,None]  # NT x 1
    t = TT.flatten()[:,None]  # NT x 1
    u = UU.flatten()[:,None]  # NT x 1
    v = VV.flatten()[:,None]  # NT x 1
    p = PP.flatten()[:,None]  # NT x 1
    
    idx = np.random.choice(N*T, N_train, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    t_train = t[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]

    training_data = [(x, y, t, u, v)]  # Assume data is already batched appropriately
    train(model, training_data, 1000, optimizer)

    # Prediction (optional)
    predictions = model.predict(torch.cat([x, y, t], dim=1))
    print(predictions)
