import torch
import scipy.io
import numpy as np
from NavierStokes_Pytorch_scratch import PhysicsInformedNN
# Load the saved model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
model = PhysicsInformedNN(layers).to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()  # Set the model to evaluation mode

# Load and prepare test data (similar to how you prepared training data)
data = scipy.io.loadmat('../Data/cylinder_nektar_wake.mat')
X_star = torch.tensor(data['X_star'], dtype=torch.float32).to(device)
t_star = torch.tensor(data['t'], dtype=torch.float32).to(device)

# Example: Select some points for testing
N = X_star.shape[0]
T = t_star.shape[0]
test_index = np.random.choice(N * T, 1000, replace=False)  # randomly pick 1000 points

XX = X_star[:, 0:1].repeat(1, T) # dimensions are N x T
YY = X_star[:, 1:2].repeat(1, T) # dimensions are N x T
TT = t_star.repeat(1, N).T # dimensions are N x T

x_test = XX.flatten()[test_index][:, None]
y_test = YY.flatten()[test_index][:, None]
t_test = TT.flatten()[test_index][:, None]

# Convert the test data to tensors with gradients enabled
x_test = torch.tensor(x_test, dtype=torch.float32, requires_grad=True).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32, requires_grad=True).to(device)
t_test = torch.tensor(t_test, dtype=torch.float32, requires_grad=True).to(device)

# Make predictions
with torch.no_grad():
    u_pred, v_pred, p_pred, f_u_pred, f_v_pred = model(x_test, y_test, t_test)

# Print some of the predictions (u and v velocity components)
print("Predicted u velocities:", u_pred[:10])
print("Predicted v velocities:", v_pred[:10])

# Optionally, compute and display any other results, such as error metrics or visualizations
# For example, if you have actual u and v values for the test set:
# u_actual = ...
# v_actual = ...
# error_u = torch.mean((u_pred - u_actual) ** 2).sqrt()  # RMSE
# error_v = torch.mean((v_pred - v_actual) ** 2).sqrt()
# print("RMSE for u: ", error_u)
# print("RMSE for v: ", error_v)
