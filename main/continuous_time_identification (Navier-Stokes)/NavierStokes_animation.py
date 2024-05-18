import numpy as np 
import scipy.io
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load the data
data = scipy.io.loadmat('../Data/cylinder_nektar_wake.mat')

# Extract data
P_star = data['p_star']  # Pressure data
t_star = data['t']       # Time data
X_star = data['X_star']  # Spatial coordinates

# Define the number of spatial and time points
N = X_star.shape[0]  # Number of spatial points
T = t_star.shape[0]  # Number of time points

# Define plotting limits
x_min, x_max = X_star[:, 0].min(), X_star[:, 0].max()
y_min, y_max = X_star[:, 1].min(), X_star[:, 1].max()
p_min, p_max = P_star.min(), P_star.max()

# Initialize the plot
dpi = 1000
fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)  # Increase figure size and DPI
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

# Function to initialize the plot
def init():
    ax.clear()
    # Find the number of unique x and y points
    nx = len(np.unique(X_star[:, 0]))
    ny = len(np.unique(X_star[:, 1]))

    # Reshape the pressure data for the first frame
    P_frame = P_star[:, 0].reshape((ny, nx))

    h = ax.imshow(P_frame, interpolation='nearest', cmap='rainbow', 
                  extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto', vmin=p_min, vmax=p_max)
    fig.colorbar(h, cax=cax)
    ax.set_title(f'Time: {t_star[0][0]:.2f}')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    return ax,

# Function to update the plot
def update(frame):
    ax.clear()
    # Find the number of unique x and y points
    nx = len(np.unique(X_star[:, 0]))
    ny = len(np.unique(X_star[:, 1]))

    # Reshape the pressure data for the current frame
    P_frame = P_star[:, frame].reshape((ny, nx))

    h = ax.imshow(P_frame, interpolation='nearest', cmap='rainbow', 
                  extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto', vmin=p_min, vmax=p_max)
    ax.set_title(f'Time: {t_star[frame][0]:.2f}')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    return ax,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=range(T), init_func=init, blit=False)

# Save animation as GIF with higher DPI
ani.save('./figures/pressure_prediction.gif', writer='pillow', dpi=dpi)

# Display the plot
plt.show()
