# Prediction (optional)
import torch 
import numpy as np
import time
import scipy.io
from NavierStokes_Pytorch_V1 import PhysicsInformedNN
from scipy.interpolate import griddata
# Load the data 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
data = scipy.io.loadmat('../Data/cylinder_nektar_wake.mat')

# Extract and process data
U_star = torch.tensor(data['U_star'], dtype=torch.float32).to(device)
P_star = torch.tensor(data['p_star'], dtype=torch.float32).to(device)
t_star = torch.tensor(data['t'], dtype=torch.float32).to(device)
X_star = torch.tensor(data['X_star'], dtype=torch.float32).to(device)


# Flatten and prepare data
N = X_star.shape[0]
T = t_star.shape[0]
print(N,T)
XX = X_star[:, 0:1].repeat(1, T) # dimensions are N x T
YY = X_star[:, 1:2].repeat(1, T) # dimensions are N x T
TT = t_star.repeat(1,N).T # dimensions are N x T



x = XX.flatten()[:, None]
y = YY.flatten()[:, None]
t = TT.flatten()[:, None]
u = U_star[:, 0, :].flatten()[:, None]
v = U_star[:, 1, :].flatten()[:, None]
    
predict = True
if predict:
    snap = 100
    x_star = XX[:, snap].unsqueeze(-1).requires_grad_(True).to(device)
    y_star = YY[:, snap].unsqueeze(-1).requires_grad_(True).to(device)
    t_star = TT[:, snap].unsqueeze(-1).requires_grad_(True).to(device)

    u_star = U_star[:, 0, snap].unsqueeze(1).to(device)
    v_star = U_star[:, 1, snap].unsqueeze(1).to(device)
    p_star = P_star[:, snap].unsqueeze(1).to(device)

    model = PhysicsInformedNN(layers).to(device)
    # First we load the model with no noise
    print("\n First run of the no noise model")
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()

   
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

    # compute the lammbda error
   
    # Retrieve the lambda values
    lambda_1_value = model.lambda_1.item()
    lambda_2_value = model.lambda_2.item()
    print("\n The estimated lambda values are")
    print(lambda_1_value, lambda_2_value)

    error_lambda_1 = abs(lambda_1_value - 1.0) * 100
    error_lambda_2 = abs(lambda_2_value - 0.01) / 0.01 * 100

    print(f'Error u: {error_u:e}')
    print(f'Error v: {error_v:e}')
    print(f'Error p: {error_p:e}')
    print(f'Error lambda_1: {error_lambda_1:e}')
    print(f'Error lambda_2: {error_lambda_2:e}')

    # Then the second run of the no noise model
    print("\n Second run of the no noise model")
    model.load_state_dict(torch.load('model2.pth', map_location=device))
    model.eval()
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
    print("\n The estimated lambda values are")
    print(lambda_1_value, lambda_2_value)

    error_lambda_1 = abs(lambda_1_value - 1.0) * 100
    error_lambda_2 = abs(lambda_2_value - 0.01) / 0.01 * 100

    print(f'Error u: {error_u:e}')
    print(f'Error v: {error_v:e}')
    print(f'Error p: {error_p:e}')
    print(f'Error lambda_1: {error_lambda_1:e}')
    print(f'Error lambda_2: {error_lambda_2:e}')

    # Then for the noisy  model with 1% noise
    print("\n First run of the 1% noise added model")
    model.load_state_dict(torch.load('model_noisy.pth', map_location=device))
    model.eval()
    # only retreive lambda values
    lambda_1_value_noisy = model.lambda_1.item()
    lambda_2_value_noisy = model.lambda_2.item()
    error_lambda_1 = abs(lambda_1_value_noisy - 1.0) * 100
    error_lambda_2 = abs(lambda_2_value_noisy - 0.01) / 0.01 * 100
    # The estimated lambda values is
    print("\n The estimated lambda values are")
    print(lambda_1_value_noisy, lambda_2_value_noisy)
    print(f'Error lambda_1: {error_lambda_1:e}')
    print(f'Error lambda_2: {error_lambda_2:e}')
    # save the results
    np.savetxt('error_lambda_1_noisy.txt', np.array([error_lambda_1]))
    np.savetxt('error_lambda_2_noisy.txt', np.array([error_lambda_2]))

    # Then for the second noisy  model with 1% noise
    print("\n Second run of the 1% noise added model")
    model.load_state_dict(torch.load('model_noisy_01.pth', map_location=device))
    
    model.eval()
    # only retreive lambda values
    lambda_1_value_noisy_01 = model.lambda_1.item()
    lambda_2_value_noisy_01 = model.lambda_2.item()
    error_lambda_1 = abs(lambda_1_value_noisy_01 - 1.0) * 100
    error_lambda_2 = abs(lambda_2_value_noisy_01 - 0.01) / 0.01 * 100
    # The estimated lambda values is
    print("\n The estimated lambda values are")
    print(lambda_1_value_noisy_01, lambda_2_value_noisy_01)
    print(f'Error lambda_1: {error_lambda_1:e}')
    print(f'Error lambda_2: {error_lambda_2:e}')

    # Then for the noisy  model with 5% noise
    print("\n First run of the 5% noise added model")
    model.load_state_dict(torch.load('model_noisy_05.pth', map_location=device))
    model.eval()
    # only retreive lambda values
    lambda_1_value_noisy_05 = model.lambda_1.item()
    lambda_2_value_noisy_05 = model.lambda_2.item()
    error_lambda_1 = abs(lambda_1_value_noisy_05 - 1.0) * 100
    error_lambda_2 = abs(lambda_2_value_noisy_05 - 0.01) / 0.01 * 100
    # The estimated lambda values is
    print("\n The estimated lambda values are")
    print(lambda_1_value_noisy_05, lambda_2_value_noisy_05)
    print(f'Error lambda_1: {error_lambda_1:e}')
    print(f'Error lambda_2: {error_lambda_2:e}')

    # Then for the first run of the noisy  model with 10% noise
    print("\n First run of the 10% noise added model")
    model.load_state_dict(torch.load('model_noisy_10.pth', map_location=device
    ))
    model.eval()
    # only retreive lambda values
    lambda_1_value_noisy_10 = model.lambda_1.item()
    lambda_2_value_noisy_10 = model.lambda_2.item()
    error_lambda_1 = abs(lambda_1_value_noisy_10 - 1.0) * 100
    error_lambda_2 = abs(lambda_2_value_noisy_10 - 0.01) / 0.01 * 100
    # The estimated lambda values is
    print("\n The estimated lambda values are")
    print(lambda_1_value_noisy_10, lambda_2_value_noisy_10)
    print(f'Error lambda_1: {error_lambda_1:e}')
    print(f'Error lambda_2: {error_lambda_2:e}')

    # Then for the first run of the noisy  model with 20% noise
    print("\n First run of the 20% noise added model")
    model.load_state_dict(torch.load('model_noisy_20.pth', map_location=device))
    model.eval()
    # only retreive lambda values
    lambda_1_value_noisy_20 = model.lambda_1.item()
    lambda_2_value_noisy_20 = model.lambda_2.item()
    error_lambda_1 = abs(lambda_1_value_noisy_20 - 1.0) * 100
    error_lambda_2 = abs(lambda_2_value_noisy_20 - 0.01) / 0.01 * 100
    # The estimated lambda values is
    print("\n The estimated lambda values are")
    print(lambda_1_value_noisy_20, lambda_2_value_noisy_20)
    print(f'Error lambda_1: {error_lambda_1:e}')
    print(f'Error lambda_2: {error_lambda_2:e}')
    
    
    # Generate the grid data
    # start by detaching the tensors
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
    X, Y = np.meshgrid(x,y)
    
    UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
    VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
    PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
    P_exact = griddata(X_star, p_star.flatten(), (X, Y), method='cubic')
    
    # Predict on all of time available to make a animation of the predicted plot 
    # Preallocate the array for storing the results
    UU_stars = np.zeros((nn, nn, T))
    VV_stars = np.zeros((nn, nn, T))
    PP_stars = np.zeros((nn, nn, T))
    P_exacts = np.zeros((nn, nn, T))
    print(T)
    predict = True
    if predict:
        for snap in range(T):
            print(snap)
            x_star = XX[:, snap].unsqueeze(-1).requires_grad_(True).to(device)
            y_star = YY[:, snap].unsqueeze(-1).requires_grad_(True).to(device)
            t_star = TT[:, snap].unsqueeze(-1).requires_grad_(True).to(device)

            u_star = U_star[:, 0, snap].unsqueeze(1).to(device)
            v_star = U_star[:, 1, snap].unsqueeze(1).to(device)
            p_star = P_star[:, snap].unsqueeze(1).to(device)

            model.load_state_dict(torch.load('model.pth', map_location=device))
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

            UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
            VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
            PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
            P_exact = griddata(X_star, p_star.cpu().flatten(), (X, Y), method='cubic')
            # Store the results
            UU_stars[:, :, snap] = UU_star
            VV_stars[:, :, snap] = VV_star
            PP_stars[:, :, snap] = PP_star
            P_exacts[:, :, snap] = P_exact

        # save the results for the predicted pressure field
        np.save('UU_stars.npy', UU_stars)
        np.save('VV_stars.npy', VV_stars)
        np.save('PP_stars.npy', PP_stars)
        np.save('P_exacts.npy', P_exacts)


