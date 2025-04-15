import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
from NN_Classes import StateSpaceSimulator
from NN_Classes import NeuralLinStateUpdate, NeuralLinOutput
from SI_Functions import compute_uncertainty, generate_input_signals, compute_covariavnce_matrix
seed=42
np.random.seed(seed)
torch.manual_seed(seed)


class SI_model:

    def __init__(self, n_x, n_u, n_y, hidden_size, dtype=torch.float32, device=torch.device("cpu"), tau_prior = 10, sigma_noise = 5e-1, modify_si=False, eps=1e-6, f_class=NeuralLinStateUpdate, g_class=NeuralLinOutput, seed=42):
        super().__init__()
        self.seed = seed
        self.n_x = n_x
        self.n_u = n_u
        self.n_y = n_y
        self.hidden_size = hidden_size
        self.f_class = f_class
        self.g_class = g_class

        torch.manual_seed(seed)
        f_xu = self.f_class(n_x=self.n_x, n_u=self.n_u, hidden_size=self.hidden_size)
        g_x  = self.g_class(n_x=self.n_x, n_y=self.n_y, hidden_size=self.hidden_size)
        nn_model = StateSpaceSimulator(f_xu, g_x)

        self.model  = nn_model
        self.dtype  = dtype
        self.device = device
        self.tau_prior = tau_prior
        self.sigma_noise = sigma_noise
        self.eps = eps
        self.modify_si = modify_si

    def nn_fit(self, u_fit, y_fit, x0, num_iter = 500, lr = 15e-3, print_freq = 100, return_loss=False):
        u_fit_torch = torch.tensor(u_fit, dtype=self.dtype)
        y_fit_torch = torch.tensor(y_fit, dtype=self.dtype)
        x0_torch    = torch.tensor(x0,    dtype=self.dtype, requires_grad=False)
        # Setup optimizer
        params_net = list( self.model.parameters())
        optimizer = optim.Adam([{'params': params_net, 'lr': lr},], lr=lr)

        LOSS_TOT = []
        LOSS_FIT = []

        # Training loop
        for itr in range(0, num_iter):

            optimizer.zero_grad()
            
            self.model.train()
            # Perform open-loop simulation
            y_sim, x_sim = self.model(x0_torch, u_fit_torch, return_x = True)

            # Compute fit loss
            err_fit = y_sim - y_fit_torch
            loss_fit = torch.mean(err_fit**2)

            # Compute trade-off loss
            loss = loss_fit

            LOSS_TOT.append(loss.item())
            LOSS_FIT.append(loss_fit.item())
            if itr % print_freq == 0:
                print(f'Iter {itr} | Fit Loss {loss_fit:.4f}')

            # Optimize
            loss.backward()
            optimizer.step()

        x_sim_fitted = x_sim.detach().numpy()
        y_sim_fitted = y_sim.detach().numpy()
        self.model_state_dict = self.model.state_dict()

        print(f'Iter {itr} | Fit Loss {loss_fit:.4f}')
        if not return_loss:
            return y_sim_fitted, x_sim_fitted
        else:
            return y_sim_fitted, x_sim_fitted, loss_fit.detach().numpy()
    
    def nn_predict(self, u_pred, x0_pred):
        u_pred_torch  = torch.tensor(u_pred,  dtype=self.dtype)
        x0_pred_torch = torch.tensor(x0_pred, dtype=self.dtype)
        self.model.load_state_dict(self.model_state_dict)
        with torch.no_grad():
            self.model.eval()
            y_pred, x_pred = self.model(x_0=x0_pred_torch, u=u_pred_torch, return_x=True)
        
        x_predicted = x_pred.detach().numpy()
        y_predicted = y_pred.detach().numpy()
        return y_predicted, x_predicted

    def compute_fit_cov(self, u_fit, x0_fit):
        f_xu = self.f_class(n_x=self.n_x, n_u=self.n_u, hidden_size=self.hidden_size)
        g_x  = self.g_class(n_x=self.n_x, n_y=self.n_y, hidden_size=self.hidden_size)
        model = StateSpaceSimulator(f_xu, g_x)
        model.load_state_dict(self.model_state_dict)

        y_fit, unc_std, H_prior, H_post, P_post, scaling_P, scaling_H, P_y = compute_covariavnce_matrix(u_fit=u_fit, x0_fit=x0_fit,f_xu=f_xu, g_x=g_x, model=model, n_x=self.n_x , n_y=self.n_y,
                                                                                                        dtype=self.dtype, device=self.device, tau_prior=self.tau_prior, sigma_noise=self.sigma_noise)
        
        self.H_prior   = H_prior
        self.H_post    = H_post
        self.P_post    = P_post
        self.scaling_P = scaling_P
        self.scaling_H = scaling_H
        self.P_y       = P_y

        if self.modify_si:
            self.suprise_index_fit = 100 * (np.sum(unc_std)/np.sum(np.abs(y_fit))) * np.abs(np.mean(y_fit)) + self.eps
            # self.suprise_index_fit = 100 * (np.sum(unc_std)/np.sum(np.abs(y_fit))) * np.mean(y_fit, axis=0)
        else:
            self.suprise_index_fit = 100 * (np.sum(unc_std)/np.sum(np.abs(y_fit)))

        self.cunc_std_fit = np.sum(unc_std)

        return y_fit, unc_std, H_prior, H_post, P_post, scaling_P, scaling_H, P_y
    
    def query(self, u_query, x0_query, return_x_query=False):
        f_xu = self.f_class(n_x=self.n_x, n_u=self.n_u, hidden_size=self.hidden_size)
        g_x  = self.g_class(n_x=self.n_x, n_y=self.n_y, hidden_size=self.hidden_size)
        model = StateSpaceSimulator(f_xu, g_x)
        model.load_state_dict(self.model_state_dict)
        
        y_query, x_query, unc_std, ppd_std = compute_uncertainty(u_query=u_query, x0_query=x0_query, P_post=self.P_post, f_xu=f_xu, g_x=g_x, model=model, n_x=self.n_x, n_y=self.n_y,
                                                        dtype=self.dtype, device=self.device, tau_prior=self.tau_prior, sigma_noise=self.sigma_noise)
        if self.modify_si:
            suprise_index = 100 * (np.sum(ppd_std)/np.sum(np.abs(y_query))) * np.abs(np.mean(y_query)) + self.eps
        else:
            suprise_index = 100 * (np.sum(ppd_std)/np.sum(np.abs(y_query)))

        cunc_std = np.sum(unc_std)
        suprise_index_norm = suprise_index/self.suprise_index_fit
        cunc_std_norm      = cunc_std/self.cunc_std_fit
        if return_x_query:
            return suprise_index_norm, cunc_std_norm, suprise_index, y_query, x_query, unc_std, ppd_std
        else:
            return suprise_index_norm, cunc_std_norm, suprise_index, y_query, unc_std, ppd_std
    
    def query_stepinput(self, t_query_list, w_query, x0_query):
        t_query, u_query = generate_input_signals(t_segments_list=t_query_list, w_values=w_query, n_u=self.n_u)
        suprise_index_norm, cunc_std_norm, suprise_index, y_query, unc_std, ppd_std = self.query(u_query=u_query, x0_query=x0_query)
        return suprise_index_norm, cunc_std_norm, suprise_index, y_query, unc_std, ppd_std, u_query, t_query
    
    def sample(self, n_smaples):
        # mu_model_parameters  = torch.tensor(utils.parameters_to_vector(self.model.parameters()))
        mu_model_parameters  = utils.parameters_to_vector(self.model.parameters()).clone().detach()
        std_model_parameters = torch.tensor(np.sqrt(np.diag(self.P_post)).reshape(-1), dtype=self.dtype)
        sample_model_dict    = {}
        for sample_idx in range(n_smaples):
            f_xu_sample = self.f_class(n_x=self.n_x, n_u=self.n_u, hidden_size=self.hidden_size)
            g_x_sample  = self.g_class(n_x=self.n_x, n_y=self.n_y, hidden_size=self.hidden_size)
            model_sample = StateSpaceSimulator(f_xu_sample, g_x_sample)
            sample_parameters = torch.normal(mu_model_parameters, std_model_parameters) 
            utils.vector_to_parameters(sample_parameters, model_sample.parameters())
            sample_model_dict[sample_idx] = model_sample

        return sample_model_dict
    

class DataScaler:

    def __init__(self, mu_u, mu_y, mu_x, std_u, std_y, std_x, mu_offset_u, mu_offset_y, mu_offset_x):
        super().__init__()
        self.mu_u = mu_u
        self.mu_y = mu_y
        self.mu_x = mu_x
        self.std_u = std_u
        self.std_y = std_y
        self.std_x = std_x
        self.mu_offset_u = mu_offset_u
        self.mu_offset_y = mu_offset_y
        self.mu_offset_x = mu_offset_x

    def normalize(self, signal_denorm, mu, std, mu_offset):
        if signal_denorm is None:
            print("Input is None -> Returing None")
            return None
        else:
            signal_norm = ((signal_denorm-mu)/std) + mu_offset
            return signal_norm
    
    def denormalize(self, signal_norm, mu, std, mu_offset):
        if signal_norm is None:
            print("Input is None -> Returing None")
            return None
        else:
            signal_denorm =  (signal_norm - mu_offset) * std + mu
            return signal_denorm
        
        
class LinearizedModel:
    def __init__(self, model, n_x, n_u, n_y):
        """
        Initializes the linearized model.
        
        Args:
            f: Function representing state dynamics x(k+1) = x(k) + f(x(k), u(k))
            g: Function representing output equation y(k) = g(x(k))
            state_dim: Dimension of state vector x
            input_dim: Dimension of input vector u
        """
        self.model = model
        self.f_xu  = self.model.f_xu
        self.g_x   = self.model.g_x
        self.n_x = n_x
        self.n_u = n_u
        self.n_y = n_y

    def statespace_rep(self, x_op, u_op):
        """
        Computes the Jacobian matrices A, B, C, D at the given operating point (x0, u0).
        """
        # # Ensure they are at least 1D
        # x_op = np.atleast_1d(x_op)
        # u_op = np.atleast_1d(u_op)

        x_op = torch.tensor(x_op, dtype=torch.float32, requires_grad=True)
        u_op = torch.tensor(u_op, dtype=torch.float32, requires_grad=True)

        # Compute Jacobians
        A_tilde = torch.autograd.functional.jacobian(lambda x: self.f_xu(x, u_op), x_op)
        B       = torch.autograd.functional.jacobian(lambda u: self.f_xu(x_op, u), u_op)
        C       = torch.autograd.functional.jacobian(lambda x: self.g_x(x), x_op)
        # D       = torch.autograd.functional.jacobian(lambda u: self.g_x(x_op, u), u_op)

        # Ensure correct shapes
        A_tilde = A_tilde.view(self.n_x, self.n_x)
        B       = B.view(self.n_x, self.n_u)
        C       = C.view(self.n_y, self.n_x)
        # D       = D.view(-1, self.n_u)

        # A = I + df/dx
        A = torch.eye(self.n_x) + A_tilde

        # Compute residual terms
        R_f = self.f_xu(x_op, u_op) - A_tilde @ x_op - B @ u_op
        R_g = self.g_x(x_op) - C @ x_op

        return A.detach().numpy(), B.detach().numpy(), C.detach().numpy(), R_f.detach().numpy(),  R_g.detach().numpy()
    
    def __call__(self, u, x0, lin_freq=1, apply_res_corr_x=True, apply_res_corr_y=True, update_res_corr_x=True, update_res_corr_y=True):
        
        T = u.shape[0]
        x = np.zeros((T, self.n_x))
        y = np.zeros((T, self.n_y))

        x[0, :] = x0
        A, B, C, R_f, R_g = self.statespace_rep(x_op=x[0, :], u_op=u[0, :])
        R_f_0, R_g_0 = R_f, R_g
        self.A, self.B, self.C, self.R_f, self.R_g = A, B, C, R_f, R_g
        y[0, :] = self.C @ x[0, :] + R_g
        
        for ind in range(T-1):
            if ind > 0 and ind%lin_freq == 0 and ind != T-2:
                A, B, C, R_f, R_g = self.statespace_rep(x_op=x[ind, :], u_op=u[ind, :])
 
                if not update_res_corr_x:
                    R_f = R_f_0
                if not update_res_corr_y:
                    R_g = R_g_0
                
                self.A, self.B, self.C, self.R_f, self.R_g = A, B, C, R_f, R_g

            y[ind + 1, :], x[ind + 1, :] = self.update(u=u[ind, :], x=x[ind, :], apply_res_corr_x=apply_res_corr_x, apply_res_corr_y=apply_res_corr_y, update_ss_rep=False)

        return y, x
    
    def update(self, u, x, apply_res_corr_x=True, apply_res_corr_y=True, update_ss_rep=False):

        if update_ss_rep:
            A, B, C, R_f, R_g = self.statespace_rep(x_op=x, u_op=u)
            self.A, self.B, self.C, self.R_f, self.R_g = A, B, C, R_f, R_g

        if apply_res_corr_x:
                x_updated = self.A @ x + self.B @ u + self.R_f
        else:
            x_updated = self.A @ x + self.B @ u

        if apply_res_corr_y:
                y_updated = self.C @ x_updated + self.R_g
        else:
            y_updated = self.C @ x_updated

        return y_updated, x_updated


            
        

        
