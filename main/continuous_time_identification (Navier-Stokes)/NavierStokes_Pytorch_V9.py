class PhysicsInformedNN(nn.Module):
    def __init__(self, layers, rho1=1.0, rho2=1.0, activation='tanh', dropout_prob=0.0):
        super(PhysicsInformedNN, self).__init__()
        self.model = nn.Sequential()
        
        # Determine the device to run on (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Choose the activation function based on the input parameter
        activation_function = nn.Tanh() if activation == 'tanh' else nn.ReLU()

        # Build the neural network
        for i in range(len(layers) - 1):
            linear_layer = nn.Linear(layers[i], layers[i + 1])
            if i < len(layers) - 2:
                # Add linear and activation layers to the model
                self.model.add_module(f"linear_{i}", linear_layer)
                self.model.add_module(f"{activation}_{i}", activation_function)
                # Add dropout layer if dropout_prob > 0
                if dropout_prob > 0.0:
                    self.model.add_module(f"dropout_{i}", nn.Dropout(dropout_prob))
            else:
                # Add the final linear layer
                self.model.add_module(f"linear_{i}", linear_layer)
            # Apply Xavier initialization to the linear layer
            init.xavier_uniform_(linear_layer.weight)
            if linear_layer.bias is not None:
                init.constant_(linear_layer.bias, 0.0)
        
        # Move the model to the appropriate device
        self.model.to(self.device)
        
        # Initialize lambda parameters
        self.lambda_1 = nn.Parameter(torch.tensor([0.0], device=self.device))
        self.lambda_2 = nn.Parameter(torch.tensor([0.0], device=self.device))

        # Store the rho parameters
        self.rho1 = rho1
        self.rho2 = rho2

    def forward(self, x, y, t):
        # Concatenate inputs and pass through the model
        cat_input = torch.cat([x, y, t], dim=1)
        output = self.model(cat_input)
        
        # Split the output into psi and pressure (p)
        psi = output[:, 0:1]
        p = output[:, 1:2]
        
        # Calculate velocity components u and v
        u = grad(psi.sum(), y, create_graph=True)[0]
        v = -grad(psi.sum(), x, create_graph=True)[0]
        
        # Calculate partial derivatives for the Navier-Stokes equations
        u_t = grad(u.sum(), t, create_graph=True)[0]
        u_x = grad(u.sum(), x, create_graph=True)[0]
        u_y = grad(u.sum(), y, create_graph=True)[0]
        u_xx = grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = grad(u_y.sum(), y, create_graph=True)[0]

        v_t = grad(v.sum(), t, create_graph=True)[0]
        v_x = grad(v.sum(), x, create_graph=True)[0]
        v_y = grad(v.sum(), y, create_graph=True)[0]
        v_xx = grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = grad(v_y.sum(), y, create_graph=True)[0]
        
        p_x = grad(p.sum(), x, create_graph=True)[0]
        p_y = grad(p.sum(), y, create_graph=True)[0]

        # Calculate the residuals of the Navier-Stokes equations
        f_u = u_t + self.lambda_1 * (u * u_x + v * u_y) + p_x - self.lambda_2 * (u_xx + u_yy)
        f_v = v_t + self.lambda_1 * (u * v_x + v * v_y) + p_y - self.lambda_2 * (v_xx + v_yy)
        
        return u, v, p , f_u, f_v

    def loss_function(self, u_true, v_true, outputs):
        # Unpack the outputs
        u_pred, v_pred, p_pred, f_u_pred, f_v_pred = outputs
        # Calculate mean squared errors for velocity and residuals
        mse_u = torch.mean((u_true - u_pred) ** 2)
        mse_v = torch.mean((v_true - v_pred) ** 2)
        mse_f_u = torch.mean(f_u_pred ** 2)
        mse_f_v = torch.mean(f_v_pred ** 2)
        # Total loss is the sum of these errors weighted by rho1 and rho2
        total_loss = self.rho1 * (mse_u + mse_v) + self.rho2 * (mse_f_u + mse_f_v)
        return total_loss, mse_u, mse_v, mse_f_u, mse_f_v

