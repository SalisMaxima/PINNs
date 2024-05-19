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

class GeneralizedPINN(nn.Module):
    """
    Generalized Physics-Informed Neural Network (PINN) class.

    Attributes:
    ----------
    model : nn.Sequential
        The neural network model.
    device : torch.device
        The device (CPU or GPU) on which the model is running.
    """
    def __init__(self, layers, activation='tanh'):
        """
        Initializes the GeneralizedPINN with the given layers and activation function.

        Parameters:
        ----------
        layers : list
            List of integers specifying the number of neurons in each layer.
        activation : str
            Activation function to use ('tanh' or 'relu').
        """
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

        Parameters:
        ----------
        inputs : tuple
            Tuple of input tensors (e.g., x, y, t).

        Returns:
        -------
        torch.Tensor
            Output of the neural network.
        """
        cat_input = torch.cat(inputs, dim=1)
        return self.model(cat_input)

    def compute_derivatives(self, output, inputs):
        """
        Computes first and second order derivatives of the output with respect to inputs.

        Parameters:
        ----------
        output : torch.Tensor
            Output tensor of the neural network.
        inputs : tuple
            Tuple of input tensors.

        Returns:
        -------
        dict
            Dictionary of derivatives.
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

        Parameters:
        ----------
        inputs : tuple
            Tuple of input tensors.
        pde_residual_func : function
            User-defined function to compute the PDE residuals.

        Returns:
        -------
        torch.Tensor
            Output of the neural network.
        list
            List of PDE residuals.
        """
        output = self.forward(*inputs)
        derivatives = self.compute_derivatives(output, inputs)
        residuals = pde_residual_func(output, derivatives, *inputs)
        return output, residuals

    def loss_function(self, true_data, predicted_data, residuals):
        """
        Computes the loss function for the PINN.

        Parameters:
        ----------
        true_data : list
            List of true output data tensors.
        predicted_data : torch.Tensor
            Predicted output tensor from the neural network.
        residuals : list
            List of PDE residuals.

        Returns:
        -------
        torch.Tensor
            Total loss.
        """
        loss = 0
        for true, pred in zip(true_data, predicted_data):
            loss += torch.mean((true - pred) ** 2)
        for res in residuals:
            loss += torch.mean(res ** 2)
        return loss

def train_pinn(data, layers, pde_residual_func, epochs=200000, batch_size=5000, lr=0.001, save_path="model.pth"):
    """
    Trains the GeneralizedPINN model using the given data and PDE residual function.

    Parameters:
    ----------
    data : dict
        Dictionary containing the input and output data for training.
    layers : list
        List of integers specifying the number of neurons in each layer.
    pde_residual_func : function
        User-defined function to compute the PDE residuals.
    epochs : int
        Number of training epochs.
    batch_size : int
        Batch size for training.
    lr : float
        Learning rate for the optimizer.
    save_path : str
        Path to save the trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess the training data
    inputs_train = [torch.tensor(data[key], dtype=torch.float32).to(device) for key in data['inputs']]
    outputs_train = [torch.tensor(data[key], dtype=torch.float32).to(device) for key in data['outputs']]
    
    model = GeneralizedPINN(layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Randomly select a batch of training data
    N_train = batch_size
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

    Parameters:
    ----------
    output : torch.Tensor
        Output tensor of the neural network.
    derivatives : dict
        Dictionary of derivatives.
    x : torch.Tensor
        Input tensor for spatial variable.
    t : torch.Tensor
        Input tensor for time variable.

    Returns:
    -------
    list
        List of PDE residuals for the Schrödinger equation.
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

if __name__ == "__main__":
    # Load data
    data = scipy.io.loadmat('../Data/NLS.mat')
    layers = [2, 100, 100, 100, 100, 2]
    save_path = "model_schrodinger.pth"
    
    # Prepare the data dictionary
    inputs = {
        'inputs': ['x', 't'],
        'outputs': ['u', 'v'],
        'x': data['x'].flatten()[:, None],
        't': data['tt'].flatten()[:, None],
        'u': np.real(data['uu']).T.flatten()[:, None],
        'v': np.imag(data['uu']).T.flatten()[:, None]
    }
    
    # Train the PINN model
    train_pinn(inputs, layers, schrodinger_pde_residual, epochs=50000, batch_size=5000, lr=0.001, save_path=save_path)
