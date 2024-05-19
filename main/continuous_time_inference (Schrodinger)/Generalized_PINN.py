import numpy as np
import torch
import torch.nn as nn
import scipy.io
import matplotlib.pyplot as plt
from torch.autograd import grad
from scipy.interpolate import griddata
from pyDOE import lhs
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set seeds for reproducibility
np.random.seed(1234)
torch.manual_seed(1234)

class GeneralizedPINN(nn.Module):
    """
    Generalized Physics-Informed Neural Network (PINN) class.
    """
    def __init__(self, layers, activation='tanh'):
        super(GeneralizedPINN, self).__init__()
        self.model = nn.Sequential()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define the activation function
        activation_function = nn.Tanh() if activation == 'tanh' else nn.ReLU()

        # Construct the neural network
        for i in range(len(layers)-1):
            self.model.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.model.add_module(f"{activation}_{i}", activation_function)
        
        self.model.to(self.device)

    def forward(self, *inputs):
        """
        Forward pass of the neural network.
        """
        cat_input = torch.cat(inputs, dim=1)
        return self.model(cat_input)

    def compute_derivatives(self, output, inputs):
        """
        Computes first and second order derivatives of the output with respect to inputs.
        """
        derivatives = {}
        for i, inp in enumerate(inputs):
            for order in range(1, 3):  # First and second derivatives
                if (order, i) not in derivatives:
                    derivatives[(order, i)] = output
                for _ in range(order):
                    derivatives[(order, i)] = grad(derivatives[(order, i)].sum(), inp, create_graph=True)[0]
        return derivatives

    def compute_residuals(self, *inputs, pde_residual_func):
        """
        Computes the residuals for the PDE based on the neural network output and its derivatives.
        """
        output = self.forward(*inputs)
        derivatives = self.compute_derivatives(output, inputs)
        residuals = pde_residual_func(output, derivatives, *inputs)
        return output, residuals

    def loss_function(self, true_data, predicted_data, residuals):
        """
        Computes the loss function for the PINN.
        """
        loss = 0
        for true, pred in zip(true_data, predicted_data):
            loss += torch.mean((true - pred) ** 2)
        for res in residuals:
            loss += torch.mean(res ** 2)
        return loss

def train_pinn(model, data, pde_residual_func, epochs=200000, batch_size=5000, lr=0.001, save_path="model.pth"):
    """
    Trains the GeneralizedPINN model using the given data and PDE residual function.
    """
    device = model.device
    
    # Load and preprocess the training data
    inputs_train = [torch.tensor(data[key], dtype=torch.float32).to(device) for key in data['inputs']]
    outputs_train = [torch.tensor(data[key], dtype=torch.float32).to(device) for key in data['outputs']]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Ensure the batch size does not exceed the number of samples
    N_train = min(batch_size, inputs_train[0].shape[0])
    idx = np.random.choice(inputs_train[0].shape[0], N_train, replace=False)
    inputs_batch = [inp[idx, :].requires_grad_(True) for inp in inputs_train]
    outputs_batch = [out[idx, :] for out in outputs_train]

    # Train the model
    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions, residuals = model.compute_residuals(*inputs_batch, pde_residual_func=pde_residual_func)
        loss = model.loss_function(outputs_batch, predictions, residuals)
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

# Example usage for the Schrödinger equation
def schrodinger_pde_residual(output, derivatives, x, t):
    """
    Computes the residuals for the Schrödinger equation.
    """
    u = output[:, 0:1]
    v = output[:, 1:2]
    u_t = derivatives[(1, 1)]
    v_t = derivatives[(1, 1)]
    u_xx = derivatives[(2, 0)]
    v_xx = derivatives[(2, 0)]
    
    f_u = u_t + 0.5 * v_xx + (u**2 + v**2) * v
    f_v = v_t - 0.5 * u_xx - (u**2 + v**2) * u
    return [f_u, f_v]

def prepare_data():
    """
    Prepares the data for the Schrödinger equation problem.
    """
    noise = 0.0        
    
    # Domain bounds
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    N0 = 50
    N_b = 50
    N_f = 20000
    layers = [2, 100, 100, 100, 100, 2]
        
    data = scipy.io.loadmat('../Data/NLS.mat')
    
    t = data['tt'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)
    
    X, T = np.meshgrid(x,t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact_u.T.flatten()[:,None]
    v_star = Exact_v.T.flatten()[:,None]
    h_star = Exact_h.T.flatten()[:,None]
    
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x,:]
    u0 = Exact_u[idx_x,0:1]
    v0 = Exact_v[idx_x,0:1]
    
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    tb = t[idx_t,:]
    
    X_f = lb + (ub-lb)*lhs(2, N_f)
    
    return x, t, X_star, u_star, v_star, h_star, X_f, x0, u0, v0, tb, lb, ub, layers

def plot_results(x, t, X_star, u_star, v_star, h_star, u_pred, v_pred, h_pred, X_f, x0, u0, v0, tb, lb, ub):
    """
    Plots the results for the Schrödinger equation problem.
    """
    X, T = np.meshgrid(x, t)
    
    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
    V_pred = griddata(X_star, v_pred.flatten(), (X, T), method='cubic')
    H_pred = griddata(X_star, h_pred.flatten(), (X, T), method='cubic')

    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    h = ax.imshow(H_pred.T, interpolation='nearest', cmap='YlGnBu', 
                  extent=[lb[1], ub[1], lb[0], ub[0]], 
                  origin='lower', aspect='auto')
    fig.colorbar(h, cax=cax)
    
    X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
    X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
    X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)
    X_u_train = np.vstack([X0, X_lb, X_ub])

    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (X_u_train.shape[0]), markersize = 4, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[75]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[100]*np.ones((2,1)), line, 'k--', linewidth = 1)
    ax.plot(t[125]*np.ones((2,1)), line, 'k--', linewidth = 1)    
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    leg = ax.legend(frameon=False, loc = 'best')
    ax.set_title('$|h(t,x)|$', fontsize = 10)
    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    for i, ti in enumerate([75, 100, 125]):
        ax = plt.subplot(gs1[0, i])
        ax.plot(x, h_star[:,ti], 'b-', linewidth=2, label='Exact')
        ax.plot(x, H_pred[ti,:], 'r--', linewidth=2, label='Prediction')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$|h(t,x)|$')
        ax.set_title('$t = %.2f$' % (t[ti]), fontsize=10)
        ax.axis('square')
        ax.set_xlim([-5.1,5.1])
        ax.set_ylim([-0.1,5.1])
        if i == 1:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), ncol=5, frameon=False)
    
    plt.show()

if __name__ == "__main__":
    # Prepare data
    x, t, X_star, u_star, v_star, h_star, X_f, x0, u0, v0, tb, lb, ub, layers = prepare_data()
    
    # Create model
    model = GeneralizedPINN(layers).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Prepare data dictionary
    data = {
        'inputs': ['x', 't'],
        'outputs': ['u', 'v'],
        'x': X_f[:, 0:1],
        't': X_f[:, 1:2],
        'u': u_star,
        'v': v_star
    }
    
    # Train the PINN model
    train_pinn(model, data, schrodinger_pde_residual, epochs=50000, batch_size=5000, lr=0.001, save_path="model_schrodinger.pth")
    
    # Predict
    X_star_torch = torch.tensor(X_star, dtype=torch.float32).to(model.device)
    u_pred, v_pred = model(X_star_torch[:, 0:1], X_star_torch[:, 1:2]).detach().cpu().numpy().T
    h_pred = np.sqrt(u_pred**2 + v_pred**2)
    
    # Plot results
    plot_results(x, t, X_star, u_star, v_star, h_star, u_pred, v_pred, h_pred, X_f, x0, u0, v0, tb, lb, ub)
