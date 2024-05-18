import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from NavierStokes_prediction import p_pred, u_pred

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

# Create a high-resolution grid
nx, ny = 200,200  # Increase these numbers for higher resolution
x_grid = np.linspace(x_min, x_max, nx)
y_grid = np.linspace(y_min, y_max, ny)
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

# Initialize the plot
dpi = 100
fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)  # Increase figure size and DPI
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

# Function to initialize the plot
def init():
    ax.clear()
    # Interpolate the pressure data to the high-resolution grid for the first frame
    P_frame = griddata(X_star, P_star[:, 0], (X_grid, Y_grid), method='cubic')

    h = ax.imshow(P_frame, interpolation='bicubic', cmap='rainbow', 
                  extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto', vmin=p_min, vmax=p_max)
    fig.colorbar(h, cax=cax)
    ax.set_title(f'Time: {t_star[0][0]:.2f}')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    return ax,

# Function to update the plot
def update(frame):
    ax.clear()
    # Interpolate the pressure data to the high-resolution grid for the current frame
    P_frame = griddata(X_star, P_star[:, frame], (X_grid, Y_grid), method='cubic')
    
    h = ax.imshow(P_frame, interpolation='bicubic', cmap='rainbow', 
                  extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto', vmin=p_min, vmax=p_max)
    if frame == 0:
        fig.colorbar(h, cax=cax)
    ax.set_title(f'Time: {t_star[frame][0]:.2f}')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    return ax,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=range(T), init_func=init,interval=80, blit=False)

# Save animation as GIF with higher DPI
ani.save('./figures/pressure_prediction.gif', writer='pillow', dpi=dpi)

# Then do the same for the predicted pressure field
# The predicted pressure is p_pred
def init2():
    ax.clear()
    # Interpolate the pressure data to the high-resolution grid for the first frame
    P_frame = griddata(X_star, p_pred[:, 0], (X_grid, Y_grid), method='cubic')

    h = ax.imshow(P_frame, interpolation='bicubic', cmap='rainbow', 
                  extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto', vmin=p_min, vmax=p_max)
    fig.colorbar(h, cax=cax)
    ax.set_title(f'Time: {t_star[0][0]:.2f}')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    return ax,

# Function to update the plot
def update2(frame):
    ax.clear()
    # Interpolate the pressure data to the high-resolution grid for the current frame
    P_frame = griddata(X_star, p_pred[:, frame], (X_grid, Y_grid), method='cubic')
    
    h = ax.imshow(P_frame, interpolation='bicubic', cmap='rainbow', 
                  extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto', vmin=p_min, vmax=p_max)
    if frame == 0:
        fig.colorbar(h, cax=cax)
    ax.set_title(f'Time: {t_star[frame][0]:.2f}')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    return ax,

# Create animation
ani = animation.FuncAnimation(fig, update, frames=range(T), init_func=init,interval=80, blit=False)

# Save animation as GIF with higher DPI
ani.save('./figures/pressure_prediction.gif', writer='pillow', dpi=dpi)



# Finally make a comparison between the exact and predicted pressure fields
# Subtract the predicted pressure field from the exact pressure field
# Thus getting the error field
PP_Errpr = P_exact - PP_star

# Initialize the plot
fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)  # Increase figure size and DPI
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

# Function to initialize the plot
def init3():
    ax.clear()
    h = ax.imshow(PP_Errpr[:, 0], interpolation='bicubic', cmap='rainbow', 
                  extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto', vmin=p_min, vmax=p_max)
    fig.colorbar(h, cax=cax)
    ax.set_title(f'Time: {t_star[0][0]:.2f}')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    return ax,

# Function to update the plot
def update3(frame):
    ax.clear()
    h = ax.imshow(PP_Errpr[:, frame], interpolation='bicubic', cmap='rainbow', 
                  extent=[x_min, x_max, y_min, y_max], origin='lower', aspect='auto', vmin=p_min, vmax=p_max)
    if frame == 0:
        fig.colorbar(h, cax=cax)
    ax.set_title(f'Time: {t_star[frame][0]:.2f}')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    return ax,

# Create animation
ani3 = animation.FuncAnimation(fig, update3, frames=range(T), init_func=init3,interval=80, blit=False)

# Save animation as GIF with higher DPI
ani3.save('./figures/pressure_prediction_error.gif', writer='pillow', dpi=dpi)


