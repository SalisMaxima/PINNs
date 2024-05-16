import scipy.io
    # Load Data
data_vort = scipy.io.loadmat('../Data/cylinder_nektar_t0_vorticity.mat')  # Loads the vorticity data from a .mat file using SciPy.

x_vort = data_vort['x']  # Extracts the x-coordinates from the data.
y_vort = data_vort['y']  # Extracts the y-coordinates from the data.
w_vort = data_vort['w']  # Extracts the vorticity values from the data.
modes = np.asscalar(data_vort['modes'])  # Retrieves the number of modes (probably related to the resolution of the simulation).
nel = np.asscalar(data_vort['nel'])  # Retrieves the number of elements in the dataset.

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