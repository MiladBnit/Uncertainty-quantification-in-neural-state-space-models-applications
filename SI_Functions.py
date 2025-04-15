import numpy as np
import torch
import torch.nn as nn
import time
from scipy.optimize import minimize

def generate_input_signals(t_segments_list, w_values, n_u):
    t_segments = list(zip(t_segments_list, t_segments_list[1:]))
    
    time_steps = np.concatenate([np.arange(t[0], t[1]) for t in t_segments])
    Tsim = time_steps.shape[0]
    u_signal = np.zeros((Tsim, n_u))
    if n_u == 1:
        u_signal[:, 0] = np.concatenate([w * np.ones(t[1] - t[0]) for t, w in zip(t_segments, w_values[:])])
    else:
        for i in range(n_u):
            u_signal[:, i] = np.concatenate([w * np.ones(t[1] - t[0]) for t, w in zip(t_segments, w_values[:, i])])
    
    return time_steps, u_signal


def generate_step_signals(t_segments_list, w_values, n_u, dt=1):
    ind_init = int(t_segments_list[0]/dt)
    ind_end = int(t_segments_list[-1]/dt)
    time_steps = np.arange(ind_init, ind_end) * dt
    Tsim = time_steps.shape[0]
    u_signal = np.zeros((Tsim, n_u))
    for k in range(w_values.shape[0]):
        for i in range(n_u):
            u_signal[(time_steps >= t_segments_list[k]), i] = w_values[k, i]
    return time_steps, u_signal


def generate_random_input_signals(low, high, n_samples, n_u, seed=None):
    if seed is not None:
        np.random.seed(seed)
    t_rand = sorted(set([int(i) for i in np.random.uniform(low=low, high=high, size=n_samples)]))
    w_rand = np.random.randn(len(t_rand) - 1)
    time_steps, u = generate_input_signals(t_rand, w_rand, n_u)
    return time_steps, u

def compute_covariavnce_matrix(u_fit, x0_fit, f_xu, g_x, model, n_x , n_y, dtype, device, tau_prior=10, sigma_noise=5e-1):
    """
    Compute covariance matrix of the given data.
    
    Parameters: 
        data (np.array): Data matrix.
    
    Returns:
        cov (np.array): Covariance matrix of the data.
    """
    N = u_fit.shape[0]
    var_noise = sigma_noise**2
    beta_noise = 1/var_noise
    var_prior = 1 / tau_prior

    n_param  = sum(map(torch.numel, model.parameters()))
    n_fparam = sum(map(torch.numel, f_xu.parameters()))
    n_gparam = sum(map(torch.numel, g_x.parameters()))

    # Evaluate the model in open-loop simulation against validation data

    u = torch.from_numpy(u_fit).to(dtype=dtype)
    # x_step = torch.zeros(n_x, dtype=dtype, requires_grad=True)
    x_step = torch.tensor(x0_fit, dtype=dtype, requires_grad=True)
    s_step = torch.zeros(n_x, n_fparam, dtype=dtype)

    scaling_H = 1/(N * beta_noise)
    scaling_P = 1/scaling_H
    # scaling_phi = np.sqrt(beta_noise * scaling_H)  # np.sqrt(1/N)

    time_start = time.time()
    # negative Hessian of the log-prior
    H_prior = torch.eye(n_param, dtype=dtype) * tau_prior * scaling_H
    P_prior = torch.eye(n_param, dtype=dtype) / tau_prior * scaling_P
    P_step  = P_prior  # prior parameter covariance
    H_step  = torch.zeros((n_param, n_param), dtype=dtype)
    #H_step = torch.eye(n_param) * beta_prior/scaling_H  # prior inverse parameter covariance

    x_sim  = []
    y_sim  = []
    J_rows = []

    basis_y = torch.eye(n_y)
    basis_x = torch.eye(n_x)

    for time_idx in range(N):
        # print(time_idx)

        # Current input
        u_step = u[time_idx, :]

        # Current state and current output sensitivity
        x_sim.append(x_step)
        y_step = g_x(x_step)
        y_sim.append(y_step)

        # Jacobian of y wrt x
        jacs_gx = [torch.autograd.grad(y_step, x_step, v, retain_graph=True)[0] for v in basis_y]
        J_gx = torch.stack(jacs_gx, dim=0)

        # Jacobian of y wrt theta
        jacs_gtheta = [torch.autograd.grad(y_step, g_x.parameters(), v, retain_graph=True) for v in basis_y]
        jacs_gtheta_f = [torch.cat([jac.ravel() for jac in jacs_gtheta[j]]) for j in range(n_y)]  # ravel jacobian rows
        J_gtheta = torch.stack(jacs_gtheta_f)  # stack jacobian rows to obtain a jacobian matrix

        # Eq. 14b in the paper "On the adaptation of recurrent neural networks for system identification",
        # Special case where f_xu and g_x are independently paremetrized
        # https://arxiv.org/abs/2201.08660
        phi_step_1 = J_gx @ s_step
        phi_step_2 = J_gtheta
        phi_step = torch.cat((phi_step_1, phi_step_2), axis=-1).t()

        J_rows.append(phi_step.t())
        H_step = H_step + phi_step @ phi_step.t() * 1/N

        den = 1 + phi_step.t() @ P_step @ phi_step
        P_tmp = - (P_step @ phi_step @ phi_step.t() @ P_step)/den
        P_step = P_step + P_tmp

        # Current x
        # System update
        delta_x = 1.0 * f_xu(x_step, u_step)

        # Jacobian of delta_x wrt x
        def get_vjp_x(v):
            return torch.autograd.grad(delta_x, x_step, v, retain_graph=True)[0]
        J_fx = torch.vmap(get_vjp_x)(basis_x)
        #jacs_fx = [torch.autograd.grad(delta_x, x_step, v, retain_graph=True)[0] for v in basis_x]
        #J_fx = torch.stack(jacs_fx, dim=0)

        # Jacobian of delta_x wrt theta
        def get_vjp_par(v):
            return torch.autograd.grad(delta_x, f_xu.parameters(), v, retain_graph=False)
        jacs_ftheta = torch.vmap(get_vjp_par)(basis_x)
        jacs_ftheta_f = [j.view(n_x, -1) for j in jacs_ftheta]
        J_ftheta = torch.cat(jacs_ftheta_f, 1)

        #jacs_ftheta = [torch.autograd.grad(delta_x, f_xu.parameters(), v, retain_graph=True) for v in basis_x]
        #jacs_ftheta_f = [torch.cat([jac.ravel() for jac in jacs_ftheta[j]]) for j in range(n_x)]  # ravel jacobian rows
        #J_ftheta = torch.stack(jacs_ftheta_f)  # stack jacobian rows to obtain a jacobian matrix

        x_step = (x_step + delta_x).detach().requires_grad_(True)

        # Eq. 14a in the paper "On the adaptation of recurrent neural networks for system identification"
        # https://arxiv.org/abs/2201.08660
        s_step = s_step + J_fx @ s_step + J_ftheta


    J = torch.cat(J_rows).squeeze(-1)
    x_sim = torch.stack(x_sim)
    y_sim = torch.stack(y_sim)

    # information matrix (or approximate negative Hessian of the log-likelihood)
    H_post = H_prior + H_step

    # time_hess = time.time()-time_start
    # print(f"GN Hessian computation time: {time_hess:.2f} s")

    P_post = torch.linalg.pinv(H_post)
    #P_step = P_step.numpy()
    #H_step = H_step.numpy()

    # Compute the covariance of the output
    P_y = J @ (P_post/scaling_P) @ J.t()

    H_prior = H_prior/scaling_H
    H_post = H_post/scaling_H
    P_post = P_post/scaling_P
    scaling_P = scaling_P
    scaling_H = scaling_H

    # Compute the standard deviation of the output
    unc_std = np.sqrt(np.diag(P_y)).reshape(-1, 1)
    y_sim = y_sim.detach().numpy()

    return y_sim, unc_std, H_prior, H_post, P_post, scaling_P, scaling_H, P_y


def compute_uncertainty(u_query, x0_query, P_post, f_xu, g_x, model, n_x , n_y, dtype, device, tau_prior=10, sigma_noise=5e-1):

    N = u_query.shape[0]
    var_noise  = sigma_noise**2
    beta_noise = 1/var_noise
    var_prior  = 1/tau_prior

    n_param = sum(map(torch.numel, model.parameters()))
    n_fparam = sum(map(torch.numel, f_xu.parameters()))
    n_gparam = sum(map(torch.numel, g_x.parameters()))

    # Evaluate the model in open-loop simulation against validation data

    u = torch.from_numpy(u_query).to(dtype=dtype)
    # x_step = torch.zeros(n_x, dtype=dtype, requires_grad=True)
    x_step = torch.tensor(x0_query, dtype=dtype, requires_grad=True)
    s_step = torch.zeros(n_x, n_fparam, dtype=dtype)

    # time_start = time.time()

    x_sim = []
    y_sim = []
    unc_var_step = []

    basis_y = torch.eye(n_y)
    basis_x = torch.eye(n_x)

    for time_idx in range(N):
        # print(time_idx)

        # Current input
        u_step = u[time_idx, :]

        # Current state and current output sensitivity
        x_sim.append(x_step)
        y_step = g_x(x_step)
        y_sim.append(y_step)

        # Jacobian of y wrt x
        jacs_gx = [torch.autograd.grad(y_step, x_step, v, retain_graph=True)[0] for v in basis_y]
        J_gx = torch.stack(jacs_gx, dim=0)

        # Jacobian of y wrt theta
        jacs_gtheta = [torch.autograd.grad(y_step, g_x.parameters(), v, retain_graph=True) for v in basis_y]
        jacs_gtheta_f = [torch.cat([jac.ravel() for jac in jacs_gtheta[j]]) for j in range(n_y)]  # ravel jacobian rows
        J_gtheta = torch.stack(jacs_gtheta_f)  # stack jacobian rows to obtain a jacobian matrix

        # Eq. 14b in the paper "On the adaptation of recurrent neural networks for system identification",
        # Special case where f_xu and g_x are independently paremetrized
        # https://arxiv.org/abs/2201.08660
        phi_step_1 = J_gx @ s_step
        phi_step_2 = J_gtheta
        phi_step = torch.cat((phi_step_1, phi_step_2), axis=-1).t()
        unc_var_step.append(phi_step.t() @ P_post @ phi_step)  # output variance at time step

        # Current x
        # System update
        delta_x = 1.0 * f_xu(x_step, u_step)

        # Jacobian of delta_x wrt x
        def get_vjp_x(v):
            return torch.autograd.grad(delta_x, x_step, v, retain_graph=True)[0]
        J_fx = torch.vmap(get_vjp_x)(basis_x)
        #jacs_fx = [torch.autograd.grad(delta_x, x_step, v, retain_graph=True)[0] for v in basis_x]
        #J_fx = torch.stack(jacs_fx, dim=0)

        # Jacobian of delta_x wrt theta
        def get_vjp_par(v):
            return torch.autograd.grad(delta_x, f_xu.parameters(), v, retain_graph=False)
        jacs_ftheta = torch.vmap(get_vjp_par)(basis_x)
        jacs_ftheta_f = [j.view(n_x, -1) for j in jacs_ftheta]
        J_ftheta = torch.cat(jacs_ftheta_f, 1)

        #jacs_ftheta = [torch.autograd.grad(delta_x, f_xu.parameters(), v, retain_graph=True) for v in basis_x]
        #jacs_ftheta_f = [torch.cat([jac.ravel() for jac in jacs_ftheta[j]]) for j in range(n_x)]  # ravel jacobian rows
        #J_ftheta = torch.stack(jacs_ftheta_f)  # stack jacobian rows to obtain a jacobian matrix

        x_step = (x_step + delta_x).detach().requires_grad_(True)

        # Eq. 14a in the paper "On the adaptation of recurrent neural networks for system identification"
        # https://arxiv.org/abs/2201.08660
        s_step = s_step + J_fx @ s_step + J_ftheta

    unc_var = torch.cat(unc_var_step)
    x_sim = torch.stack(x_sim)
    y_sim = torch.stack(y_sim)

    # time_hess = time.time()-time_start
    # print(f"Query computation time: {time_hess:.2f} s")

    y_sim = y_sim.detach().numpy()
    x_sim = x_sim.detach().numpy()
    unc_var = unc_var.detach().numpy()
    unc_std = np.sqrt(unc_var).reshape(-1, 1)

    ppd_var = unc_var + var_noise
    ppd_std = np.sqrt(ppd_var)

    return y_sim, x_sim, unc_std, ppd_std


def perform_experiment(sys, u, x0, datascaler, normal_input=True, norm_output=True):
    if normal_input:
        u_norm = u
        x0_norm = x0
        u_denorm  = datascaler.denormalize(u_norm,  datascaler.mu_u, datascaler.std_u, datascaler.mu_offset_u)
        x0_denorm = datascaler.denormalize(x0_norm, datascaler.mu_u, datascaler.std_u, datascaler.mu_offset_u)
    else:
        u_denorm = u
        x0_denorm = x0
    
    y_denorm, x_denorm = sys(u_denorm, x0_denorm)

    if norm_output:
        y_norm = datascaler.normalize(y_denorm, datascaler.mu_y, datascaler.std_y, datascaler.mu_offset_y)
        x_norm = datascaler.normalize(x_denorm, datascaler.mu_x, datascaler.std_x, datascaler.mu_offset_x)
        y = y_norm
        x = x_norm
    else:
        y = y_denorm
        x = x_denorm
        
    return y, x



def rNMPC(si_model, y_ref, x0, u0, Wy, Wu, Wunc, Np_rh, dt=1, sampling_interval=1, u_min=None, u_max=None, x_min=None, x_max=None, y_min=None, y_max=None, ppd_std_max=None, unc_index_norm_max=None, unc_index_w=1, method='SLSQP', warm_start=False, warm_start_method='SLSQP', apply_warm_start_constraints=False, ig_w_rand=0, ig_warm_start_w_rand=0):
    """
    Solve Nonlinear MPC problem.
    """
    n_x = si_model.n_x
    n_u = si_model.n_u
    n_y = si_model.n_y

    def cost_function(u_control):
        """Objective function for NMPC."""
        u_control = u_control.reshape((Np_rh, n_u))
        t_control = np.arange(0, Np_rh+1)*sampling_interval
        
        _, u = generate_step_signals(t_segments_list=t_control, w_values=u_control, n_u=n_u, dt=dt)
        suprise_index_norm, cunc_std_norm, suprise_index, y, x, unc_std, ppd_std = si_model.query(u, x0, return_x_query=True)
        unc_index_norm = unc_index_w * cunc_std_norm + (1-unc_index_w) * suprise_index_norm
        cost = 0
        for k in range(y.shape[0]):
            cost = cost + ((y[k, :] - y_ref[k, :]) @ Wy @ (y[k, :] - y_ref[k, :]).T)
            cost = cost + unc_index_norm**2 * Wunc
            if k > 0:
                cost = cost + ((u[k, :]-u[k-1, :]) @ Wu @ (u[k, :]-u[k-1, :]).T)
        return cost

    def constraint_function(u_control):
        """Inequality constraints on states and outputs."""
        u_control = u_control.reshape((Np_rh, n_u))
        t_control = np.arange(0, Np_rh+1)*sampling_interval
        _, u = generate_step_signals(t_segments_list=t_control, w_values=u_control, n_u=n_u, dt=dt)
        suprise_index_norm, cunc_std_norm, suprise_index, y, x, unc_std, ppd_std = si_model.query(u, x0, return_x_query=True)
        unc_index_norm = unc_index_w * cunc_std_norm + (1-unc_index_w) * suprise_index_norm
        constraints = []
        for k in range(y.shape[0]):
            if y_max is not None:
                for i in range(n_y):
                    constraints.append(y_max[0, i] - y[k, i])  
            if y_min is not None:
                for i in range(n_y):
                    constraints.append(y[k, i] - y_min[0, i])
            if x_max is not None:
                for i in range(n_x):
                    constraints.append(x_max[0, i] - x[k, i])  
            if x_min is not None:
                for i in range(n_x):
                    constraints.append(x[k, i] - x_min[0, i])
            
            if ppd_std_max is not None:
                for i in range(n_y):
                    constraints.append(ppd_std_max[0, i] - ppd_std[k, i])

            if unc_index_norm_max is not None:
                constraints.append(unc_index_norm_max - unc_index_norm)

        return np.array(constraints).flatten()

    if u_min is not None and u_max is not None:
        if isinstance(u_min, (int, float)):
            u_bounds = [(u_min, u_max)] * (Np_rh * n_u)  # Scalar case: Apply same bounds to all inputs
        else:
            u_bounds = [(u_min[0, i], u_max[0, i]) for i in range(n_u)] * Np_rh 

    constraints = {'type': 'ineq', 'fun': constraint_function}

    options = {'maxiter': 1000}

    u0_control = (1-ig_warm_start_w_rand) * u0[:Np_rh, :] + ig_warm_start_w_rand * np.random.randn(Np_rh, n_u) * np.max(np.abs(u0), axis=0)

    if warm_start:
        if apply_warm_start_constraints:
            if all(const is None for const in [y_max, y_min, x_max, x_min, ppd_std_max, unc_index_norm_max]):
                if u_min is None and u_max is None:
                    result_warm_start = minimize(cost_function, u0_control.reshape(Np_rh*n_u, ), method=warm_start_method)
                else:
                    result_warm_start = minimize(cost_function, u0_control.reshape(Np_rh*n_u, ), method=warm_start_method, bounds=u_bounds)
            else:
                if u_min is None and u_max is None:
                    result_warm_start = minimize(cost_function, u0_control.reshape(Np_rh*n_u, ), method=warm_start_method, constraints=constraints)
                else:
                    result_warm_start = minimize(cost_function, u0_control.reshape(Np_rh*n_u, ), method=warm_start_method, constraints=constraints, bounds=u_bounds)
        else:
            result_warm_start = minimize(cost_function, u0_control.reshape(Np_rh*n_u, ), method=warm_start_method)
        u0_control = result_warm_start.x.reshape((Np_rh, n_u))

    u0_control = (1-ig_w_rand) * u0_control[:Np_rh, :] + ig_w_rand * np.random.randn(Np_rh, n_u) * np.max(np.abs(u0_control), axis=0)

    if all(const is None for const in [y_max, y_min, x_max, x_min, ppd_std_max, unc_index_norm_max]):
        if u_min is None and u_max is None:
            result = minimize(cost_function, u0_control.reshape(Np_rh*n_u, ), method=method, options=options)
        else:
            result = minimize(cost_function, u0_control.reshape(Np_rh*n_u, ), method=method, bounds=u_bounds, options=options)
    else:
        if u_min is None and u_max is None:
            result = minimize(cost_function, u0_control.reshape(Np_rh*n_u, ), method=method, constraints=constraints, options=options)
        else:
            result = minimize(cost_function, u0_control.reshape(Np_rh*n_u, ), method=method, constraints=constraints, bounds=u_bounds, options=options)

    if result.success:
        info = 'Solver converged successfully'
    else:
        info = 'Solver failed to converge'

    u_control_opt = result.x.reshape((Np_rh, n_u))
    t_control = np.arange(0, Np_rh+1)*sampling_interval
    _, u_opt = generate_step_signals(t_segments_list=t_control, w_values=u_control_opt, n_u=n_u, dt=dt)
    suprise_index_norm, cunc_std_norm, suprise_index, y_opt, x_opt, unc_std, ppd_std = si_model.query(u_opt, x0, return_x_query=True)
    unc_index_norm = unc_index_w * cunc_std_norm + (1-unc_index_w) * suprise_index_norm

    return  y_opt, u_opt, x_opt, info, ppd_std, unc_index_norm