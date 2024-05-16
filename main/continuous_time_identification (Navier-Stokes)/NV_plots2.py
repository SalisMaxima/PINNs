# Use Agg backend for matplotlib
import matplotlib
matplotlib.use('Agg')

# Import necessary libraries
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from plotting import newfig, savefig
from NavierStokes_prediction import *
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D

# Define auxiliary functions for plotting
def axisEqual3D(ax): # From NavierStokes.py by Raisii et al
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

# Set seeds for reproducibility of experiments
np.random.seed(1234)

# Load the data
data = scipy.io.loadmat('../Data/cylinder_nektar_wake.mat')

# Extract and process data
U_star = data['U_star']
P_star = data['p_star']
t_star = data['t']
X_star = data['X_star']

# Convert tensors to NumPy arrays and detach them if needed
x_star = X_star[:, 0:1].copy()
y_star = X_star[:, 1:2].copy()

# Flatten and prepare data
N = X_star.shape[0]
T = t_star.shape[0]
XX = np.tile(x_star, (1, T)) # dimensions are N x T
YY = np.tile(y_star, (1, T)) # dimensions are N x T
TT = np.tile(t_star.T, (N, 1)) # dimensions are N x T

x = XX.flatten()[:, None]
y = YY.flatten()[:, None]
t = TT.flatten()[:, None]
u = U_star[:, 0, :].flatten()[:, None]
v = U_star[:, 1, :].flatten()[:, None]

# Select training data
N_train = 5000
idx = np.random.choice(N * T, N_train, replace=False)
x_train = x[idx, :]
y_train = y[idx, :]
t_train = t[idx, :]
u_train = u[idx, :]
v_train = v[idx, :]

# Load Data
data_vort = scipy.io.loadmat('../Data/cylinder_nektar_t0_vorticity.mat')  # Loads the vorticity data from a .mat file using SciPy.

x_vort = data_vort['x']  # Extracts the x-coordinates from the data.
y_vort = data_vort['y']  # Extracts the y-coordinates from the data.
w_vort = data_vort['w']  # Extracts the vorticity values from the data.
modes = data_vort['modes'].item()  # Retrieves the number of modes
nel = data_vort['nel'].item()  # Retrieves the number of elements in the dataset

# Reshape the coordinates and vorticity arrays to match the grid structure in the data.
xx_vort = np.reshape(x_vort, (modes+1, modes+1, nel), order='F')
yy_vort = np.reshape(y_vort, (modes+1, modes+1, nel), order='F')
ww_vort = np.reshape(w_vort, (modes+1, modes+1, nel), order='F')

# Set the plot limits.
box_lb = np.array([1.0, -2.0])  # Lower bounds of the plot box.
box_ub = np.array([8.0, 2.0])  # Upper bounds of the plot box.

# Initialize the plot with custom dimensions.
fig, ax = newfig(1.0, 1.2)
ax.axis('off')

# Create a grid for the plot.
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-2/4 + 0.12, left=0.0, right=1.0, wspace=0)
ax = plt.subplot(gs0[:, :])

# Plot the vorticity for each element in the dataset.
for i in range(0, nel):
    h = ax.pcolormesh(xx_vort[:,:,i], yy_vort[:,:,i], ww_vort[:,:,i], cmap='seismic', shading='gouraud', vmin=-3, vmax=3)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(h, cax=cax)  # Adds a colorbar to the plot.

# Draw a bounding box around the data.
ax.plot([box_lb[0], box_lb[0]], [box_lb[1], box_ub[1]], 'k', linewidth=1)
ax.plot([box_ub[0], box_ub[0]], [box_lb[1], box_ub[1]], 'k', linewidth=1)
ax.plot([box_lb[0], box_ub[0]], [box_lb[1], box_lb[1]], 'k', linewidth=1)
ax.plot([box_lb[0], box_ub[0]], [box_ub[1], box_ub[1]], 'k', linewidth=1)

# Set aspect ratio, labels, and title for the plot.
ax.set_aspect('equal', 'box')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_title('Vorticity', fontsize=10)

# Display the plot.
# plt.show()

####### Row 1: Training data ##################
########      u(t,x,y)     ###################
# Create a GridSpec object with 1 row and 2 columns
gs1 = gridspec.GridSpec(1, 2)
# Update GridSpec layout parameters
gs1.update(top=1-2/4, bottom=0.0, left=0.01, right=0.99, wspace=0)
# Create a subplot with 3D projection in the first GridSpec slot
ax = plt.subplot(gs1[:, 0], projection='3d')
# Turn off axis lines and labels
ax.axis('off')

# Define ranges for x, t, and y coordinates
r1 = [x_star.min(), x_star.max()]
r2 = [t_star.min(), t_star.max()]
r3 = [y_star.min(), y_star.max()]

# Plot 3D box edges based on coordinate ranges
for s, e in combinations(np.array(list(product(r1, r2, r3))), 2):
    if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
        ax.plot3D(*zip(s, e), color="k", linewidth=0.5, zorder=1)  # Lower zorder for box edges

# Scatter plot of training data points
ax.scatter(x_train, t_train, y_train, s=0.1, zorder=2)  # Lower zorder for scatter plot

# Contour plot of training data
ax.contourf(XX, U_star[:, 0, :], YY, zdir='y', offset=t_star.mean(), cmap='rainbow', alpha=0.8, zorder=3)  # Higher zorder for contour plot

# Label the axes
ax.text(x_star.mean(), t_star.min() - 1, y_star.min() - 1, '$x$', zorder=4)
ax.text(x_star.max() + 1, t_star.mean(), y_star.min() - 1, '$t$', zorder=4)
ax.text(x_star.min() - 1, t_star.min() - 0.5, y_star.mean(), '$y$', zorder=4)
ax.text(x_star.min() - 3, t_star.mean(), y_star.max() + 1, '$u(t,x,y)$', zorder=4)

# Set the limits for the 3D plot
ax.set_xlim3d(r1)
ax.set_ylim3d(r2)
ax.set_zlim3d(r3)
# Ensure the aspect ratio is equal
axisEqual3D(ax)

########      v(t,x,y)     ###################
# Create a subplot with 3D projection in the second GridSpec slot
ax = plt.subplot(gs1[:, 1], projection='3d')
# Turn off axis lines and labels
ax.axis('off')

# Define ranges for x, t, and y coordinates
r1 = [x_star.min(), x_star.max()]
r2 = [t_star.min(), t_star.max()]
r3 = [y_star.min(), y_star.max()]

# Plot 3D box edges based on coordinate ranges
for s, e in combinations(np.array(list(product(r1, r2, r3))), 2):
    if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
        ax.plot3D(*zip(s, e), color="k", linewidth=0.5, zorder=1)  # Lower zorder for box edges

# Scatter plot of training data points
ax.scatter(x_train, t_train, y_train, s=0.1, zorder=2)  # Lower zorder for scatter plot

# Contour plot of training data
ax.contourf(XX, U_star[:, 1, :], YY, zdir='y', offset=t_star.mean(), cmap='rainbow', alpha=0.8, zorder=3)  # Higher zorder for contour plot

# Label the axes
ax.text(x_star.mean(), t_star.min() - 1, y_star.min() - 1, '$x$', zorder=4)
ax.text(x_star.max() + 1, t_star.mean(), y_star.min() - 1, '$t$', zorder=4)
ax.text(x_star.min() - 1, t_star.min() - 0.5, y_star.mean(), '$y$', zorder=4)
ax.text(x_star.min() - 3, t_star.mean(), y_star.max() + 1, '$v(t,x,y)$', zorder=4)

# Set the limits for the 3D plot
ax.set_xlim3d(r1)
ax.set_ylim3d(r2)
ax.set_zlim3d(r3)
# Ensure the aspect ratio is equal
axisEqual3D(ax)

# Save figure (commented out)
savefig('./figures/NavierStokes_data')

# Create a new figure and axis
fig, ax = newfig(1.015, 0.8)
# Turn off axis lines and labels
ax.axis('off')

######## Row 2: Pressure #######################
########      Predicted p(t,x,y)     ###########
# Create a GridSpec object with 1 row and 2 columns
gs2 = gridspec.GridSpec(1, 2)
# Update GridSpec layout parameters
gs2.update(top=1, bottom=1-1/2, left=0.1, right=0.9, wspace=0.5)
# Create a subplot in the first GridSpec slot
ax = plt.subplot(gs2[:, 0])
# Display an image of the predicted pressure
h = ax.imshow(PP_star, interpolation='nearest', cmap='rainbow', 
              extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()], 
              origin='lower', aspect='auto')
# Create a colorbar for the image
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

# Add the colorbar to the figure
fig.colorbar(h, cax=cax)
# Label the axes
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
# Ensure the aspect ratio is equal
ax.set_aspect('equal', 'box')
# Set the title of the plot
ax.set_title('Predicted pressure', fontsize=10)

########     Exact p(t,x,y)     ###########
# Create a subplot in the second GridSpec slot
ax = plt.subplot(gs2[:, 1])
# Display an image of the exact pressure
h = ax.imshow(P_exact, interpolation='nearest', cmap='rainbow', 
              extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()], 
              origin='lower', aspect='auto')
# Create a colorbar for the image
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

# Add the colorbar to the figure
fig.colorbar(h, cax=cax)
# Label the axes
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
# Ensure the aspect ratio is equal
ax.set_aspect('equal', 'box')
# Set the title of the plot
ax.set_title('Exact pressure', fontsize=10)

######## Row 3: Table #######################
# Create a GridSpec object with 1 row and 2 columns
gs3 = gridspec.GridSpec(1, 2)
# Update GridSpec layout parameters
gs3.update(top=1-1/2, bottom=0.0, left=0.0, right=1.0, wspace=0)
# Create a subplot using the entire GridSpec layout
ax = plt.subplot(gs3[:, :])
# Turn off axis lines and labels
ax.axis('off')

# Ensure lambda values are defined
lambda_1_value = 0.1  # Replace with your actual value
lambda_2_value = 0.01  # Replace with your actual value
lambda_1_value_noisy = 0.1  # Replace with your actual value
lambda_2_value_noisy = 0.01  # Replace with your actual value

# Define a string for the table in LaTeX format
s = r'$\begin{tabular}{|c|c|}';
s = s + r' \hline'
s = s + r' Correct PDE & $\begin{array}{c}'
s = s + r' u_t + (u u_x + v u_y) = -p_x + 0.01 (u_{xx} + u_{yy})\\'
s = s + r' v_t + (u v_x + v v_y) = -p_y + 0.01 (v_{xx} + v_{yy})'
s = s + r' \end{array}$ \\ '
s = s + r' \hline'
s = s + r' Identified PDE (clean data) & $\begin{array}{c}'
s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value, lambda_2_value)
s = s + r' \\'
s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value, lambda_2_value)
s = s + r' \end{array}$ \\ '
s = s + r' \hline'
s = s + r' Identified PDE (1\% noise) & $\begin{array}{c}'
s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
s = s + r' \\'
s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value_noisy, lambda_2_value_noisy)
s = s + r' \end{array}$ \\ '
s = s + r' \hline'
s = s + r' \end{tabular}$'

ax.text(0.015, 0.0, s)

# Save figure (commented out)
plt.savefig('./figures/NavierStokes_prediction.png')  # Save as PNG
