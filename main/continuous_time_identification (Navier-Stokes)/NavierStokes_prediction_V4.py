import torch
import numpy as np
import pandas as pd
import scipy.io
from scipy.interpolate import griddata
from NavierStokes_Pytorch_V1 import PhysicsInformedNN
from NV_plots_V3 import plot_vorticity_and_training_data

def evaluate_models(data_path, vorticity_path, models_paths, layers, produce_table=False, prepare_for_animations=False, prepare_for_plots=False, evaluate_models=True):
    """
    Evaluate multiple models for predicting the Navier-Stokes equations.
‰
    Args:
    - data_path (str): Path to the .mat file containing the data.
    - vorticity_path (str): Path to the .mat file containing the vorticity data.
    - models_paths (list): List of paths to the models to be evaluated.
    - layers (list): List defining the structure of the neural network layers.
    - produce_table (bool): If True, produces a table of lambda values and errors, and saves it as a CSV file.
    - prepare_for_animations (bool): If True, prepares the velocity and pressure fields for all time steps for animations.
    - prepare_for_plots (bool): If True along with evaluate models, prepares data for the plots and calls the plotting function.
    - evaluate_models (bool): If True, evaluates the models, calculates errors, and stores the lambda values and their errors.

    Returns:
    - None
    """
    
    # Set device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data from the specified .mat file
    data = scipy.io.loadmat(data_path)

    # Extract and process data
    U_star = torch.tensor(data['U_star'], dtype=torch.float32).to(device)
    P_star = torch.tensor(data['p_star'], dtype=torch.float32).to(device)
    t_star = torch.tensor(data['t'], dtype=torch.float32).to(device)
    X_star = torch.tensor(data['X_star'], dtype=torch.float32).to(device)

    # Flatten and prepare data
    N = X_star.shape[0]
    T = t_star.shape[0]
    XX = X_star[:, 0:1].repeat(1, T)  # dimensions are N x T
    YY = X_star[:, 1:2].repeat(1, T)  # dimensions are N x T
    TT = t_star.repeat(1, N).T  # dimensions are N x T

    x = XX.flatten()[:, None]
    y = YY.flatten()[:, None]
    t = TT.flatten()[:, None]
    u = U_star[:, 0, :].flatten()[:, None]
    v = U_star[:, 1, :].flatten()[:, None]

    results = []
    lambda_values = {}
    
    if evaluate_models:
        for i, model_path in enumerate(models_paths):
            noise_level = ["No noise", "1% noise", "5% noise", "10% noise"][i]
            print(f"\n {noise_level} model")
            snap = 100
            x_star = XX[:, snap].unsqueeze(-1).requires_grad_(True).to(device)
            y_star = YY[:, snap].unsqueeze(-1).requires_grad_(True).to(device)
            t_star = TT[:, snap].unsqueeze(-1).requires_grad_(True).to(device)

            u_star = U_star[:, 0, snap].unsqueeze(1).to(device)
            v_star = U_star[:, 1, snap].unsqueeze(1).to(device)
            p_star = P_star[:, snap].unsqueeze(1).to(device)

            # Initialize and load the model
            model = PhysicsInformedNN(layers).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            # Predict using the model
            u_pred, v_pred, p_pred, f_u_pred, f_v_pred = model(x_star, y_star, t_star)
            u_pred = u_pred.detach()
            v_pred = v_pred.detach()
            p_pred = p_pred.detach()
            f_u_pred = f_u_pred.detach()
            f_v_pred = f_v_pred.detach()

            # Compute the error with 2 norm
            error_u = (u_star - u_pred).norm(2) / u_star.norm(2)
            error_v = (v_star - v_pred).norm(2) / v_star.norm(2)
            error_p = (p_star - p_pred).norm(2) / p_star.norm(2)
            
            # Retrieve the lambda values
            lambda_1_value = model.lambda_1.item()
            lambda_2_value = model.lambda_2.item()

            print(f"\n The estimated lambda values are: {lambda_1_value}, {lambda_2_value}")

            error_lambda_1 = abs(lambda_1_value - 1.0) * 100
            error_lambda_2 = abs(lambda_2_value - 0.01) / 0.01 * 100

            print(f'Error u: {error_u:e}')
            print(f'Error v: {error_v:e}')
            print(f'Error p: {error_p:e}')
            print(f'Error lambda_1: {error_lambda_1:e}')
            print(f'Error lambda_2: {error_lambda_2:e}')
            
            results.append([noise_level, lambda_1_value, lambda_2_value, error_lambda_1, error_lambda_2])
            lambda_values[noise_level] = {'lambda_1': lambda_1_value, 'lambda_2': lambda_2_value}

    if produce_table:
        # Create a DataFrame and save to CSV
        df_lambda = pd.DataFrame(results, columns=['Model', 'Lambda_1', 'Lambda_2', 'Error_Lambda_1 (%)', 'Error_Lambda_2 (%)'])
        print(df_lambda)
        df_lambda.to_csv('lambda_values_errors.csv', index=False)

    if prepare_for_plots and evaluate_models:
        # Generate the grid data for plotting
        snap = 100
        x_star = XX[:, snap].unsqueeze(-1).requires_grad_(True).to(device)
        y_star = YY[:, snap].unsqueeze(-1).requires_grad_(True).to(device)
        t_star = TT[:, snap].unsqueeze(-1).requires_grad_(True).to(device)

        u_star = U_star[:, 0, snap].unsqueeze(1).to(device)
        v_star = U_star[:, 1, snap].unsqueeze(1).to(device)
        p_star = P_star[:, snap].unsqueeze(1).to(device)

        # Initialize and load the model
        model = PhysicsInformedNN(layers).to(device)
        model.load_state_dict(torch.load(models_paths[0], map_location=device)) # Assuming the first model is used for plotting. This should be the no noise model.
        model.eval()
        
        # Predict using the model
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = model(x_star, y_star, t_star)
        
        # Detach tensors and convert to numpy arrays
        u_pred = u_pred.cpu().detach().numpy()
        v_pred = v_pred.cpu().detach().numpy()
        p_pred = p_pred.cpu().detach().numpy()
        p_star = p_star.cpu().detach().numpy()
        X_star = X_star.cpu().detach().numpy()
        x_star = x_star.cpu().detach().numpy()
        y_star = y_star.cpu().detach().numpy()
        t_star = t_star.cpu().detach().numpy()

        # Generate the grid data
        lb = X_star.min(0)
        ub = X_star.max(0)
        nn = 200
        x = np.linspace(lb[0], ub[0], nn)
        y = np.linspace(lb[1], ub[1], nn)
        X, Y = np.meshgrid(x, y)

        UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
        VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
        PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
        P_exact = griddata(X_star, p_star.flatten(), (X, Y), method='cubic')

        # Call the plotting function
        plot_vorticity_and_training_data(data_path, vorticity_path, lambda_values,X,Y, UU_star, VV_star, PP_star, P_exact)

    if prepare_for_animations:
        # Predict on all time steps to prepare data for animations
        lb = X_star.min(0)
        ub = X_star.max(0)
        nn = 200
        x = np.linspace(lb[0], ub[0], nn)
        y = np.linspace(lb[1], ub[1], nn)
        X, Y = np.meshgrid(x, y)

        UU_stars = np.zeros((nn, nn, T))
        VV_stars = np.zeros((nn, nn, T))
        PP_stars = np.zeros((nn, nn, T))
        P_exacts = np.zeros((nn, nn, T))

        for snap in range(T):
            print(snap)
            x_star = XX[:, snap].unsqueeze(-1).requires_grad_(True).to(device)
            y_star = YY[:, snap].unsqueeze(-1).requires_grad_(True).to(device)
            t_star = TT[:, snap].unsqueeze(-1).requires_grad_(True).to(device)

            u_star = U_star[:, 0, snap].unsqueeze(1).to(device)
            v_star = U_star[:, 1, snap].unsqueeze(1).to(device)
            p_star = P_star[:, snap].unsqueeze(1).to(device)

            # Load the model (assuming the first model is used for prediction on all time steps)
            model.load_state_dict(torch.load(models_paths[0], map_location=device))
            model.eval()

            u_pred, v_pred, p_pred, f_u_pred, f_v_pred = model(x_star, y_star, t_star)
            u_pred = u_pred.detach()
            v_pred = v_pred.detach()
            p_pred = p_pred.detach()
            f_u_pred = f_u_pred.detach()
            f_v_pred = f_v_pred.detach()

            # Generate the grid data
            u_pred = u_pred.cpu().detach().numpy()
            v_pred = v_pred.cpu().detach().numpy()
            p_pred = p_pred.cpu().detach().numpy()

            UU_star = griddata(X_star.cpu().detach().numpy(), u_pred.flatten(), (X, Y), method='cubic')
            VV_star = griddata(X_star.cpu().detach().numpy(), v_pred.flatten(), (X, Y), method='cubic')
            PP_star = griddata(X_star.cpu().detach().numpy(), p_pred.flatten(), (X, Y), method='cubic')
            P_exact = griddata(X_star.cpu().detach().numpy(), p_star.cpu().flatten(), (X, Y), method='cubic')

            # Store the results
            UU_stars[:, :, snap] = UU_star
            VV_stars[:, :, snap] = VV_star
            PP_stars[:, :, snap] = PP_star
            P_exacts[:, :, snap] = P_exact

        # Save the results for the predicted fields
        np.save('UU_stars.npy', UU_stars)
        np.save('VV_stars.npy', VV_stars)
        np.save('PP_stars.npy', PP_stars)
        np.save('P_exacts.npy', P_exacts)


# Example usage of the function
data_path = '../Data/cylinder_nektar_wake.mat'
vorticity_path = '../Data/cylinder_nektar_t0_vorticity.mat'
models_paths = ['model_xavier.pth', 'model_noisy_xavier_01.pth', 'model_noisy_xavier_05.pth', 'model_noisy_xavier_10.pth']
layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]

# Call the function with desired parameters
evaluate_models(data_path, vorticity_path, models_paths, layers, produce_table=True, prepare_for_animations=False, prepare_for_plots=True, evaluate_models=True)
