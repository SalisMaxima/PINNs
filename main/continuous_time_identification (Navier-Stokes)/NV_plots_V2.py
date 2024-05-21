import scipy.io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from plotting import newfig, savefig
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from NavierStokes_prediction_V3 import evaluate_models
# Define auxiliary functions for plotting
def axisEqual3D(ax):  # From NavierStokes.py by Raisii et al
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)

# Define the function for plotting vorticity and training data
def plot_vorticity_and_training_data(data_path, vorticity_path, lambda_values):
    """
    Plots vorticity and training data for the Navier-Stokes equations.

    Args:
    - data_path (str): Path to the .mat file containing the data.
    - vorticity_path (str): Path to the .mat file containing the vorticity data.
    - lambda_values (dict): Dictionary containing lambda values for different noise levels.
    """

    # Set seeds for reproducibility of experiments
    np.random.seed(1234)

    # Load the data
    data = scipy.io.loadmat(data_path)

    # Extract and process data
    U_star = data['U_star']
    P_star = data['p_star']
    t_star = data['t']
    X_star = data['X_star']

    # Flatten and prepare data
    N = X_star.shape[0]
    T = t_star.shape[0]
    XX = np.tile(X_star[:, 0:1], (1, T))  # dimensions are N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # dimensions are N x T
    TT = np.tile(t_star.T, (N, 1))  # dimensions are N x T

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

    # Load vorticity data
    data_vort = scipy.io.loadmat(vorticity_path)

    # Extract vorticity data
    x_vort = data_vort['x']
    y_vort = data_vort['y']
    w_vort = data_vort['w']
    modes = data_vort['modes'].item()
    nel = data_vort['nel'].item()

    # Reshape the coordinates and vorticity arrays
    xx_vort = np.reshape(x_vort, (modes + 1, modes + 1, nel), order='F')
    yy_vort = np.reshape(y_vort, (modes + 1, modes + 1, nel), order='F')
    ww_vort = np.reshape(w_vort, (modes + 1, modes + 1, nel), order='F')

    # Set the plot limits
    box_lb = np.array([1.0, -2.0])  # Lower bounds of the plot box
    box_ub = np.array([8.0, 2.0])  # Upper bounds of the plot box

    # Initialize the plot with custom dimensions
    fig, ax = newfig(1.0, 1.2)
    ax.axis('off')

    # Create a grid for the plot
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 2 / 4 + 0.12, left=0.0, right=1.0, wspace=0)
    ax = plt.subplot(gs0[:, :])

    # Plot the vorticity for each element in the dataset
    for i in range(0, nel):
        h = ax.pcolormesh(xx_vort[:, :, i], yy_vort[:, :, i], ww_vort[:, :, i], cmap='seismic', shading='gouraud', vmin=-3, vmax=3)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)  # Adds a colorbar to the plot

    # Draw a bounding box around the data
    ax.plot([box_lb[0], box_lb[0]], [box_lb[1], box_ub[1]], 'k', linewidth=1)
    ax.plot([box_ub[0], box_ub[0]], [box_lb[1], box_ub[1]], 'k', linewidth=1)
    ax.plot([box_lb[0], box_ub[0]], [box_lb[1], box_lb[1]], 'k', linewidth=1)
    ax.plot([box_lb[0], box_ub[0]], [box_ub[1], box_ub[1]], 'k', linewidth=1)

    # Set aspect ratio, labels, and title for the plot
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Vorticity', fontsize=10)

    # Create a GridSpec object with 1 row and 2 columns
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1 - 2 / 4, bottom=0.0, left=0.01, right=0.99, wspace=0.1)

    ########      u(t,x,y)     ###################
    # Create a subplot with 3D projection in the first GridSpec slot
    ax = plt.subplot(gs1[:, 0], projection='3d')
    ax.axis('off')

    # Define ranges for x, t, and y coordinates
    r1 = [X_star[:, 0].min(), X_star[:, 0].max()]
    r2 = [t_star.min(), t_star.max()]
    r3 = [X_star[:, 1].min(), X_star[:, 1].max()]

    # Plot 3D box edges based on coordinate ranges
    for s, e in combinations(np.array(list(product(r1, r2, r3))), 2):
        if np.sum(np.abs(s - e)) == r1[1] - r1[0] or np.sum(np.abs(s - e)) == r2[1] - r2[0] or np.sum(np.abs(s - e)) == r3[1] - r3[0]:
            ax.plot3D(*zip(s, e), color="k", linewidth=0.5)

    # Scatter plot of training data points
    ax.scatter(x_train, t_train, y_train, s=0.1, alpha=0.15)
    # Contour plot of training data (Note: UU_star should be provided as input)
    ax.contourf(X_star[:, 0], UU_star, X_star[:, 1], zdir='y', offset=t_star.mean(), cmap='rainbow', alpha=0.85)

    # Label the axes
    ax.text(X_star[:, 0].mean(), t_star.min() - 1, X_star[:, 1].min() - 1, '$x$')
    ax.text(X_star[:, 0].max() + 1, t_star.mean(), X_star[:, 1].min() - 1, '$t$')
    ax.text(X_star[:, 0].min() - 1, t_star.min() - 0.5, X_star[:, 1].mean(), '$y$')
    ax.text(X_star[:, 0].min() - 3, t_star.mean(), X_star[:, 1].max() + 1, '$u(t,x,y)$')

    # Set the limits for the 3D plot
    ax.set_xlim3d(r1)
    ax.set_ylim3d(r2)
    ax.set_zlim3d(r3)
    axisEqual3D(ax)

    ########      v(t,x,y)     ###################
    # Create a subplot with 3D projection in the second GridSpec slot
    ax = plt.subplot(gs1[:, 1], projection='3d')
    ax.axis('off')

    # Plot 3D box edges based on coordinate ranges
    for s, e in combinations(np.array(list(product(r1, r2, r3))), 2):
        if np.sum(np.abs(s - e)) == r1[1] - r1[0] or np.sum(np.abs(s - e)) == r2[1] - r2[0] or np.sum(np.abs(s - e)) == r3[1] - r3[0]:
            ax.plot3D(*zip(s, e), color="k", linewidth=0.5)

    # Scatter plot of training data points
    ax.scatter(x_train, t_train, y_train, s=0.1, alpha=0.15)
    # Contour plot of training data (Note: VV_star should be provided as input)
    ax.contourf(X_star[:, 0], VV_star, X_star[:, 1], zdir='y', offset=t_star.mean(), cmap='rainbow', alpha=0.85)

    # Label the axes
    ax.text(X_star[:, 0].mean(), t_star.min() - 1, X_star[:, 1].min() - 1, '$x$')
    ax.text(X_star[:, 0].max() + 1, t_star.mean(), X_star[:, 1].min() - 1, '$t$')
    ax.text(X_star[:, 0].min() - 1, t_star.min() - 0.5, X_star[:, 1].mean(), '$y$')
    ax.text(X_star[:, 0].min() - 3, t_star.mean(), X_star[:, 1].max() + 1, '$v(t,x,y)$')

    # Set the limits for the 3D plot
    ax.set_xlim3d(r1)
    ax.set_ylim3d(r2)
    ax.set_zlim3d(r3)
    axisEqual3D(ax)

    # Save figure
    savefig('./figures/NavierStokes_data_my2', crop=False)

    # Create a new figure and axis
    fig, ax = newfig(1.015, 0.8)
    ax.axis('off')

    ######## Row 2: Pressure #######################
    ########      Predicted p(t,x,y)     ###########
    # Create a GridSpec object with 1 row and 2 columns
    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=1, bottom=1 - 1 / 2, left=0.1, right=0.9, wspace=0.5)
    # Create a subplot in the first GridSpec slot
    ax = plt.subplot(gs2[:, 0])
    # Display an image of the predicted pressure (Note: PP_star should be provided as input)
    h = ax.imshow(PP_star, interpolation='nearest', cmap='rainbow',
                  extent=[X_star[:, 0].min(), X_star[:, 0].max(), X_star[:, 1].min(), X_star[:, 1].max()],
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
    # Display an image of the exact pressure (Note: P_exact should be provided as input)
    h = ax.imshow(P_exact, interpolation='nearest', cmap='rainbow',
                  extent=[X_star[:, 0].min(), X_star[:, 0].max(), X_star[:, 1].min(), X_star[:, 1].max()],
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
    gs3.update(top=1 - 1 / 2, bottom=0.0, left=0.0, right=1.0, wspace=0)
    # Create a subplot using the entire GridSpec layout
    ax = plt.subplot(gs3[:, :])
    ax.axis('off')

    # Define a string for the table in LaTeX format
    s = r'$\begin{tabular}{|c|c|}';
    s = s + r' \hline'
    s = s + r' Correct PDE & $\begin{array}{c}'
    s = s + r' u_t + (u u_x + v u_y) = -p_x + 0.01 (u_{xx} + u_{yy})\\'
    s = s + r' v_t + (u v_x + v v_y) = -p_y + 0.01 (v_{xx} + v_{yy})'
    s = s + r' \end{array}$ \\ '
    s = s + r' \hline'
    s = s + r' Identified PDE (clean data) & $\begin{array}{c}'
    s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_values['clean']['lambda_1'], lambda_values['clean']['lambda_2'])
    s = s + r' \\'
    s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_values['clean']['lambda_1'], lambda_values['clean']['lambda_2'])
    s = s + r' \end{array}$ \\ '
    s = s + r' \hline'
    s = s + r' Identified PDE (1\% noise) & $\begin{array}{c}'
    s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_values['noisy_01']['lambda_1'], lambda_values['noisy_01']['lambda_2'])
    s = s + r' \\'
    s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_values['noisy_01']['lambda_1'], lambda_values['noisy_01']['lambda_2'])
    s = s + r' \end{array}$ \\ '
    s = s + r' \hline'
    s = s + r' Identified PDE (5\% noise) & $\begin{array}{c}'
    s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_values['noisy_05']['lambda_1'], lambda_values['noisy_05']['lambda_2'])
    s = s + r' \\'  
    s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_values['noisy_05']['lambda_1'], lambda_values['noisy_05']['lambda_2'])
    s = s + r' \end{array}$ \\ '
    s = s + r' \hline'
    s = s + r' \end{tabular}$'

    ax.text(0.015, 0.0, s)

    # Save figure
    savefig('./figures/NavierStokes_prediction')
# Get the lambda values for different noise levels using the evaluate_models function
data_path = '../Data/cylinder_nektar_wake.mat'
models_paths = ['model_xavier.pth', 'model_noisy_xavier_01.pth', 'model_noisy_xavier_05.pth', 'model_noisy_xavier_10.pth']
layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
evaluate_models(data_path, models_paths, layers, produce_table=False, prepare_for_animations=False, prepare_for_plots=True, evaluate_models=False)

data_path = '../Data/cylinder_nektar_wake.mat'
vorticity_path = '../Data/cylinder_nektar_t0_vorticity.mat'
plot_vorticity_and_training_data(data_path, vorticity_path, lambda_values)
