# Prediction (optional)
import torch 
import numpy as np
import time
import scipy.io
from NavierStokes_Pytorch_V1 import PhysicsInformedNN
# Load the data 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
data = scipy.io.loadmat('../Data/cylinder_nektar_wake.mat')

# Extract and process data
U_star = torch.tensor(data['U_star'], dtype=torch.float32).to(device)
# print the size of U_star in bytes
#print(U_star.element_size() * U_star.nelement())
# print the size of U_star in GB
#print(U_star.element_size() * U_star.nelement() / 1024**3)
P_star = torch.tensor(data['p_star'], dtype=torch.float32).to(device)
# print the size of p_star in bytes
#print(P_star.element_size() * P_star.nelement())
# print the size of p_star in GB
#print(P_star.element_size() * P_star.nelement() / 1024**3)
t_star = torch.tensor(data['t'], dtype=torch.float32).to(device)
# print the size of t_star in bytes
#print(t_star.element_size() * t_star.nelement())
# print the size of t_star in GB
#print(t_star.element_size() * t_star.nelement() / 1024**3)
X_star = torch.tensor(data['X_star'], dtype=torch.float32).to(device)
# print the size of X_star in bytes
#print(X_star.element_size() * X_star.nelement())
# print the size of X_star in GB
#print(X_star.element_size() * X_star.nelement() / 1024**3)

# Flatten and prepare data
N = X_star.shape[0]
T = t_star.shape[0]
print(N,T)
XX = X_star[:, 0:1].repeat(1, T) # dimensions are N x T
YY = X_star[:, 1:2].repeat(1, T) # dimensions are N x T
TT = t_star.repeat(1,N).T # dimensions are N x T
print(XX.shape,YY.shape,TT.shape)
# wait for 5 seconds
time.sleep(1)

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
    model.load_state_dict(torch.load('model.pth', map_location=device))
    model.eval()

   
    u_pred, v_pred, p_pred, f_u_pred, f_v_pred = model(x_star, y_star, t_star)
    u_pred = u_pred.detach()
    v_pred = v_pred.detach()
    p_pred = p_pred.detach()
    f_u_pred = f_u_pred.detach()
    f_v_pred = f_v_pred.detach()


    # Compute the error with np.linalg.norm
    error_u = (u_star - u_pred).norm(2) / u_star.norm(2)
    error_v = (v_star - v_pred).norm(2) / v_star.norm(2)
    error_p = (p_star - p_pred).norm(2) / p_star.norm(2)

    # compute the lammbda error
    # error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
    # error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100
    # Retrieve the lambda values
    lambda_1_value = model.lambda_1.item()
    lambda_2_value = model.lambda_2.item()

    print(lambda_1_value, lambda_2_value)

    error_lambda_1 = abs(lambda_1_value - 1.0) * 100
    error_lambda_2 = abs(lambda_2_value - 0.01) / 0.01 * 100

    print(f'Error u: {error_u:e}')
    print(f'Error v: {error_v:e}')
    print(f'Error p: {error_p:e}')
    print(f'Error lambda_1: {error_lambda_1:e}')
    print(f'Error lambda_2: {error_lambda_2:e}')


    # save the results
    np.savetxt('u_pred.txt', u_pred.cpu().numpy())
    np.savetxt('v_pred.txt', v_pred.cpu().numpy())
    np.savetxt('p_pred.txt', p_pred.cpu().numpy())
    np.savetxt('f_u_pred.txt', f_u_pred.cpu().numpy())
    np.savetxt('f_v_pred.txt', f_v_pred.cpu().numpy())
    np.savetxt('error_u.txt', np.array([error_u.cpu().numpy()]))
    np.savetxt('error_v.txt', np.array([error_v.cpu().numpy()]))
    np.savetxt('error_p.txt', np.array([error_p.cpu().numpy()]))
    np.savetxt('error_lambda_1.txt', np.array([error_lambda_1]))
    np.savetxt('error_lambda_2.txt', np.array([error_lambda_2]))
    # also the lambda values
    np.savetxt('lambda_1.txt', np.array([lambda_1_value]))
    np.savetxt('lambda_2.txt', np.array([lambda_2_value]))

    print('Results saved')



    # Then for the hoisy  model
    model.load_state_dict(torch.load('model_noisy.pth', map_location=device))
    model.eval()
    # only retreive lambda values
    lambda_1_value = model.lambda_1.item()
    lambda_2_value = model.lambda_2.item()
    error_lambda_1 = abs(lambda_1_value - 1.0) * 100
    error_lambda_2 = abs(lambda_2_value - 0.01) / 0.01 * 100
    print(f'Error lambda_1: {error_lambda_1:e}')
    print(f'Error lambda_2: {error_lambda_2:e}')
    # save the results
    np.savetxt('error_lambda_1_noisy.txt', np.array([error_lambda_1]))