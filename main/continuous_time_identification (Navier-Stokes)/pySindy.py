import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pysindy as ps
import torch

# Load the data from the .mat file
data_path = '../Data/cylinder_nektar_wake.mat'
data = loadmat(data_path)

# Extract and process data
U_star = data['U_star']  # N x 2 x T array
P_star = data['p_star']  # N x T array
t_star = np.ravel(data['t'])  # T array
X_star = data['X_star']  # N x 2 array

N = X_star.shape[0]
T = t_star.shape[0]
XX = X_star[:, 0:1].repeat(T, axis=1)
YY = X_star[:, 1:2].repeat(T, axis=1)
TT = t_star.reshape(1, -1).repeat(N, axis=0)

x = XX.flatten()
y = YY.flatten()
t = TT.flatten()
u = U_star[:, 0, :].flatten()
v = U_star[:, 1, :].flatten()

# Reshape u and v to the appropriate format for pySindy
u = u.reshape(N, T, 1)
v = v.reshape(N, T, 1)

# Function to plot u and its time derivative u_dot (if needed for visualization)
def plot_u_and_u_dot(t, x, u):
    plt.figure(figsize=(12, 6))

    # Plot u
    plt.subplot(121)
    plt.pcolor(t, x, u[:, :, 0], shading='auto')
    plt.title('u')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.colorbar()

    # Compute u_dot using finite differences
    u_dot = np.gradient(u, t, axis=1)

    # Plot u_dot
    plt.subplot(122)
    plt.pcolor(t, x, u_dot[:, :, 0], shading='auto')
    plt.title('u_dot')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.colorbar()

    plt.show()

    return u_dot

# Call the plot function (optional)
u_dot = plot_u_and_u_dot(t_star, X_star[:, 0], u)

# Define the PDE library that is quadratic in u, and fourth-order in spatial derivatives of u
library_functions = [lambda x: x, lambda x: x * x]
library_function_names = [lambda x: x, lambda x: x + x]

pde_lib = ps.PDELibrary(
    library_functions=library_functions,
    function_names=library_function_names,
    derivative_order=4,
    spatial_grid=X_star[:, 0],
    is_uniform=True,
)

# Define and fit the SINDy model
print('SR3 model: ')
optimizer = ps.SR3(threshold=30, normalize_columns=True)
model = ps.SINDy(feature_library=pde_lib, feature_names=['u'], optimizer=optimizer)
model.fit(u, t=np.diff(t_star).mean())
model.print()
