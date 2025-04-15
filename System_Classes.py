import numpy as np
seed=42
np.random.seed(seed)


class LTISystem:
    def __init__(self, A, B, C, add_noise=False, noise_magnitude = 0):
        super().__init__()
        # Model-reference matrices
        self.A = A
        self.B = B
        self.C = C
        self.n_x = A.shape[0]
        self.n_u = B.shape[1]
        self.n_y = C.shape[0]
        self.add_noise = add_noise
        self.noise_magnitude = noise_magnitude

    def __call__(self, u, x0):

        if len(u.shape) == 1:
            print(f"u.shape: {u.shape} -> returning y0, x0 as the output")
            u = u.reshape(1, -1)

        T = u.shape[0]
        x = np.zeros((T, self.n_x))
        y = np.zeros((T, self.n_y))
        x[0, :] = x0
        y[0, :] = self.C @ x0
        for ind in range(T-1):
            y[ind + 1, :], x[ind + 1, :] = self.update(u=u[ind, :], x=x[ind, :])
        if not self.add_noise:
            return y, x
        else:
            y_noisy = y + self.noise_magnitude * np.random.randn(T, self.n_y)
            return y_noisy, x
        
    def update(self, u, x):
        x_updated = self.A @ x + self.B @ u
        y_updated = self.C @ x_updated

        if not self.add_noise:
            return y_updated, x_updated
        else:
            y_noisy = y_updated + self.noise_magnitude * np.random.randn(self.n_y)
            return y_noisy, x_updated


class WHSystem:
    def __init__(self, A_1, B_1, C_1, f_nonlin, A_2, B_2, C_2, add_noise=False, noise_magnitude = 0):
        super().__init__()
        # Model-reference matrices
        self.A_1 = A_1
        self.B_1 = B_1
        self.C_1 = C_1
        self.A_2 = A_2
        self.B_2 = B_2
        self.C_2 = C_2
        self.f_nonlin = f_nonlin
        self.n_x_1 = A_1.shape[0]
        self.n_u_1 = B_1.shape[1]
        self.n_y_1 = C_1.shape[0]
        self.n_x_2 = A_2.shape[0]
        self.n_u_2 = B_2.shape[1]
        self.n_y_2 = C_2.shape[0]
        self.add_noise = add_noise
        self.noise_magnitude = noise_magnitude
        self.n_x = self.n_x_2
        self.n_y = self.n_y_2
        self.n_u = self.n_u_1

    def __call__(self, u, x0):
        x0_1 = x0
        x0_2 = x0

        if len(u.shape) == 1:
            print(f"u.shape: {u.shape} -> returning y0, x0 as the output")
            u = u.reshape(1, -1)

        T = u.shape[0]
        x_1 = np.zeros((T, self.n_x_1))
        y_1 = np.zeros((T, self.n_y_1))
        x_1[0, :] = x0_1
        y_1[0, :] = self.C_1 @ x0_1
        x_2 = np.zeros((T, self.n_x_2))
        y_2 = np.zeros((T, self.n_y_2))
        x_2[0, :] = x0_2
        y_2[0, :] = self.C_2 @ x0_2
        for ind in range(T-1):
            y_1[ind + 1, :], x_1[ind + 1, :], y_2[ind + 1, :], x_2[ind + 1, :] = self.update(u=u[ind, :], x_1=x_1[ind, :], x_2=x_2[ind, :])
        x = x_2
        y = y_2

        if not self.add_noise:
            return y, x
        else:
            y_noisy = y + self.noise_magnitude * np.random.randn(T, self.n_y_2)
            return y_noisy, x
    
    def update(self, u, x_1, x_2):
        y_1 = self.C_1 @ x_1
        fy_1 = self.f_nonlin(y_1)
        x_1_updated = self.A_1 @ x_1 + self.B_1 @ u
        y_1_updated = self.C_1 @ x_1_updated
        x_2_updated = self.A_1 @ x_2 + self.B_2 @ fy_1
        y_2_updated = self.C_1 @ x_2_updated
        
        return y_1_updated, x_1_updated, y_2_updated, x_2_updated


class CascadedTankSystem:

    def __init__(self, min_cons_hard_x_step=0, max_cons_hard_x_step=10, min_cons_hard_y_step=0, max_cons_hard_y_step=10, tau_u=0.25, tau_y=0.25, dt=1, add_noise=False, noise_magnitude=0, return_SNR=False):
        super().__init__()
        self.add_noise = add_noise
        self.noise_magnitude = noise_magnitude
        self.return_SNR = return_SNR
        self.min_cons_hard_x_step = min_cons_hard_x_step
        self.max_cons_hard_x_step = max_cons_hard_x_step
        self.min_cons_hard_y_step = min_cons_hard_y_step
        self.max_cons_hard_y_step = max_cons_hard_y_step
        self.n_x = 4
        self.n_y = 1
        self.n_u = 1
        self.tau_u = tau_u
        self.tau_y = tau_y
        self.dt = dt

    def f(self, x_step, u_ref, theta_f=[0.1925, 0.2429, 0.1697, 0.0444, 0.1642]):
        
        tau_u = self.tau_u
        tau_y = self.tau_y

        x_1 = x_step[0:1]
        x_2 = x_step[1:2]
        x_3 = x_step[2:3]
        u   = x_step[3:4]

        theta_f_1 = theta_f[0]
        theta_f_3 = theta_f[1]
        theta_f_4 = theta_f[2]
        theta_f_5 = theta_f[3]
        theta_f_6 = theta_f[4]

        du = (tau_u - 1) * u + (1 - tau_u) * u_ref

        dx_1 = -theta_f_1 * np.sqrt(x_1) + theta_f_5 / 10 * x_1 + theta_f_4 * u

        x_1_overflow = x_1 + dx_1
        if x_1_overflow <= 10:
            dx_2 = theta_f_1 * np.sqrt(x_1) - theta_f_5 / 10 * x_1 + theta_f_6 / 10 * x_2 - theta_f_3 * np.sqrt(x_2)
        else:
            dx_2 = (theta_f_1 * np.sqrt(x_1) - theta_f_5 / 10 * x_1 + 
                    theta_f_6 / 10 * x_2 - theta_f_3 * np.sqrt(x_2) + theta_f_5 * u)

        dx_3 = (tau_y - 1) * x_3 + (1 - tau_y) * x_2

        dx = np.concat([dx_1, dx_2, dx_3, du])  # Use np.array instead of np.stack

        return dx

    def h(self, x_step):
        return np.array([x_step[2:3]])  # Ensuring output is array-shaped

    def __call__(self, u, x0):

        if len(u.shape) == 1:
            print(f"u.shape: {u.shape} -> returning y0, x0 as the output")
            u = u.reshape(1, -1)

        T = u.shape[0]
        x = np.zeros((T, self.n_x))
        y = np.zeros((T, self.n_y))
        noise = np.zeros((T, 1))

        min_cons_hard_x_step = self.min_cons_hard_x_step
        max_cons_hard_x_step = self.max_cons_hard_x_step
        min_cons_hard_y_step = self.min_cons_hard_y_step
        max_cons_hard_y_step = self.max_cons_hard_y_step

        x[0, :] = x0
        y[0, :] = self.h(x_step=x0)
        for ind in range(T-1):
            if self.return_SNR:
                y[ind + 1, :], x[ind + 1, :], noise[ind + 1, :] = self.update(u=u[ind, :], x=x[ind, :])
            else:
                y[ind + 1, :], x[ind + 1, :] = self.update(u=u[ind, :], x=x[ind, :])
            # Imposing Constraints
            x[ind + 1, :] = np.clip(x[ind + 1, :], min_cons_hard_x_step, max_cons_hard_x_step)
            y[ind + 1, :] = np.clip(y[ind + 1, :], min_cons_hard_y_step, max_cons_hard_y_step)

        if self.return_SNR:
            num = np.sum(np.square(y[1:, :] - y[:-1, :]), axis=0)
            den = np.sum(np.square(noise[1:, :] - noise[:-1, :]), axis=0)
            SNR = 10 * np.log10(num / den)
            print("SNR: ", SNR)

        return y, x
    
    def update(self, u, x):
    
        x_updated = x + self.dt * self.f(x_step=x, u_ref=u)
        y_updated = self.h(x_step=x_updated)
    
        if not self.add_noise:
                return y_updated, x_updated
        else:
            noise = self.noise_magnitude * np.random.randn(self.n_y)
            y_noisy = y_updated + noise
            if self.return_SNR:
                return y_noisy, x_updated, noise
            else:
                return y_noisy, x_updated


class CascadedTankSystem_Simplified:

    def __init__(self, min_cons_hard_x_step=0, max_cons_hard_x_step=10, min_cons_hard_y_step=0, max_cons_hard_y_step=10, dt=1):
        super().__init__()
        self.min_cons_hard_x_step = min_cons_hard_x_step
        self.max_cons_hard_x_step = max_cons_hard_x_step
        self.min_cons_hard_y_step = min_cons_hard_y_step
        self.max_cons_hard_y_step = max_cons_hard_y_step
        self.n_x = 2
        self.n_y = 1
        self.n_u = 1
        self.dt  = dt

    def f(self, x_step, u, theta_f=[0.1925, 0.2429, 0.1697, 0.0444, 0.1642]):

        x_1 = x_step[0:1]
        x_2 = x_step[1:2]

        theta_f_1 = theta_f[0]
        theta_f_3 = theta_f[1]
        theta_f_4 = theta_f[2]
        theta_f_5 = theta_f[3]
        theta_f_6 = theta_f[4]

        dx_1 = -theta_f_1 * np.sqrt(x_1) + theta_f_5 / 10 * x_1 + theta_f_4 * u

        x_1_overflow = x_1 + dx_1
        if x_1_overflow <= 10:
            dx_2 = theta_f_1 * np.sqrt(x_1) - theta_f_5 / 10 * x_1 + theta_f_6 / 10 * x_2 - theta_f_3 * np.sqrt(x_2)
        else:
            dx_2 = (theta_f_1 * np.sqrt(x_1) - theta_f_5 / 10 * x_1 + 
                    theta_f_6 / 10 * x_2 - theta_f_3 * np.sqrt(x_2) + theta_f_5 * u)

        dx = np.concat([dx_1, dx_2])  # Use np.array instead of np.stack

        return dx

    def h(self, x_step):
        return np.array([x_step[1:2]])  # Ensuring output is array-shaped

    def __call__(self, u, x0):

        if len(u.shape) == 1:
            print(f"u.shape: {u.shape} -> returning y0, x0 as the output")
            u = u.reshape(1, -1)

        T = u.shape[0]
        x = np.zeros((T, self.n_x))
        y = np.zeros((T, self.n_y))

        min_cons_hard_x_step = self.min_cons_hard_x_step
        max_cons_hard_x_step = self.max_cons_hard_x_step
        min_cons_hard_y_step = self.min_cons_hard_y_step
        max_cons_hard_y_step = self.max_cons_hard_y_step

        x[0, :] = x0
        y[0, :] = self.h(x_step=x0)
        for ind in range(T-1):
            y[ind + 1, :], x[ind + 1, :] = self.update(u=u[ind, :], x=x[ind, :])
            # Imposing Constraints
            x[ind + 1, :] = np.clip(x[ind + 1, :], min_cons_hard_x_step, max_cons_hard_x_step)
            y[ind + 1, :] = np.clip(y[ind + 1, :], min_cons_hard_y_step, max_cons_hard_y_step)

        return y, x
    
    def update(self, u, x):
        x_updated = x + self.dt * self.f(x_step=x, u=u)
        y_updated = self.h(x_step=x_updated)
        return y_updated, x_updated
    

class CSTR_MISO_3by1:

    def __init__(self, dt=0.1):

        self.dt = dt
        # Model Parameters
        self.F        = 1
        self.V        = 1
        self.R        = 1.985875
        self.deltaH   = -5969
        self.E        = 11843
        self.k_0      = 34930800
        self.rhoC_p   = 500
        self.UA       = 500
        self.C_Af_nom = 10
        self.C_A0_nom = 8.5698
        self.T_0_nom  = 311.2639
        self.T_f_nom  = 300
        self.T_c_nom  = 292

        self.n_x = 2
        self.n_y = 1
        self.n_u = 3
    
    def f(self, x_step, u_step):

        C_A  = x_step[0:1]
        T    = x_step[1:2]
        C_Af = u_step[0:1]
        T_f  = u_step[0:1]
        T_c  = u_step[0:1]

        r = self.k_0 * np.exp(-self.E/(self.R * T)) * C_A
        dC_A = (self.F/self.V) * (C_Af - C_A) - r
        dT   = (self.F/self.V) * (T_f - T) - (self.deltaH/self.rhoC_p) * r - self.UA/(self.rhoC_p) * (T - T_c)

        dx = np.concat([dC_A, dT])

        return dx
    
    def h(self, x_step):
        return np.array([x_step[0:1]])
    
    def __call__(self, u, x0):

        if len(u.shape) == 1:
            print(f"u.shape: {u.shape} -> returning y0, x0 as the output")
            u = u.reshape(1, -1)

        T = u.shape[0]
        x = np.zeros((T, self.n_x))
        y = np.zeros((T, self.n_y))

        x[0, :] = x0
        y[0, :] = self.h(x_step=x0)
        for ind in range(T-1):
            y[ind + 1, :], x[ind + 1, :] = self.update(u=u[ind, :], x=x[ind, :])

        return y, x
    
    def update(self, u, x):
        x_updated = x + self.dt * self.f(x_step=x, u=u)
        y_updated = self.h(x_step=x_updated)
        return y_updated, x_updated
    

class CSTR_SISO:

    def __init__(self, dt=0.1):

        self.dt = dt
        # Model Parameters
        self.F        = 1
        self.V        = 1
        self.R        = 1.985875
        self.deltaH   = -5969
        self.E        = 11843
        self.k_0      = 34930800
        self.rhoC_p   = 500
        self.UA       = 150
        self.C_Af_nom = 10
        self.C_A0_nom = 8.5698
        self.T_0_nom  = 311.2639
        self.T_f_nom  = 300
        self.T_c_nom  = 292

        self.n_x = 1
        self.n_y = 1
        self.n_u = 1

        self.x0  = np.array([self.C_A0_nom])
    
    def f(self, x_step, u):

        C_A  = x_step[0:1]
        C_Af = u[0:1]
        T = self.T_0_nom

        r = self.k_0 * np.exp(-self.E/(self.R * T)) * C_A
        dC_A = (self.F/self.V) * (C_Af - C_A) - r
        # dT   = (self.F/self.V) * (T_f - T) - (self.deltaH/self.rhoC_p) * r - self.UA/(self.UA * self.V) * (T - T_c)

        dx = np.concat([dC_A])

        return dx
    
    def h(self, x_step):
        return np.array([x_step[0:1]])
    
    def __call__(self, u, x0):

        if len(u.shape) == 1:
            print(f"u.shape: {u.shape} -> returning y0, x0 as the output")
            u = u.reshape(1, -1)

        T = u.shape[0]
        x = np.zeros((T, self.n_x))
        y = np.zeros((T, self.n_y))

        x[0, :] = x0
        y[0, :] = self.h(x_step=x0)
        for ind in range(T-1):
            y[ind + 1, :], x[ind + 1, :] = self.update(u=u[ind, :], x=x[ind, :])

        return y, x
    
    def update(self, u, x):
        x_updated = x + self.dt * self.f(x_step=x, u=u)
        y_updated = self.h(x_step=x_updated)
        return y_updated, x_updated


class CSTR_MISO_2by1:

    def __init__(self, dt=0.1, add_noise=False, noise_magnitude=0):

        self.dt = dt
        self.add_noise = add_noise
        self.noise_magnitude = noise_magnitude
        # Model Parameters
        self.F        = 1
        self.V        = 1
        self.R        = 1.985875
        self.deltaH   = -5969
        self.E        = 11843
        self.k_0      = 34930800
        self.rhoC_p   = 500
        self.UA       = 500
        # self.C_Af_nom = 10
        self.C_A0_nom = 8.5698
        self.T_0_nom  = 311.2639
        self.T_f_nom  = 300
        self.T_c_nom  = 292

        self.n_x = 2
        self.n_y = 1
        self.n_u = 2

        self.x0  = np.array([self.C_A0_nom, self.T_0_nom])
    
    def f(self, x_step, u):

        C_A  = x_step[0:1]
        T    = x_step[1:2]
        C_Af = u[0:1]
        T_c  = u[1:2]

        T_f = self.T_f_nom

        r = self.k_0 * np.exp(-self.E/(self.R * T)) * C_A
        dC_A = (self.F/self.V) * (C_Af - C_A) - r
        dT   = (self.F/self.V) * (T_f - T) - (self.deltaH/self.rhoC_p) * r - self.UA/(self.rhoC_p) * (T - T_c)
        dx = np.concat([dC_A, dT])

        return dx
    
    def h(self, x_step):
        return np.array([x_step[0:1]])
    
    def __call__(self, u, x0):

        if len(u.shape) == 1:
            print(f"u.shape: {u.shape} -> returning y0, x0 as the output")
            u = u.reshape(1, -1)

        T = u.shape[0]
        x = np.zeros((T, self.n_x))
        y = np.zeros((T, self.n_y))

        x[0, :] = x0
        y[0, :] = self.h(x_step=x0)
        for ind in range(T-1):
            y[ind + 1, :], x[ind + 1, :] = self.update(u=u[ind, :], x=x[ind, :])

        return y, x
    
    def update(self, u, x):
        x_updated = x + self.dt * self.f(x_step=x, u=u)
        y_updated = self.h(x_step=x_updated)
    
        if not self.add_noise:
                return y_updated, x_updated
        else:
            y_noisy = y_updated + self.noise_magnitude * np.random.randn(self.n_y)
            return y_noisy, x_updated

    

class InvertaseProduction_FedBacthBioReactor:

    def __init__(self, dt=0.1, mu_max=0.2, Ks=0.2, Y_xs=0.5, Sf=50, F=0.02, V0=1.0, 
                 S0=10, C0=0.1, P0=0.1, alpha=50, beta=2, t_max=15):

        self.dt = dt
        self.mu_max = mu_max  # Maximum specific growth rate (1/h)
        self.Ks = Ks  # Monod constant (g/L)
        self.Y_xs = Y_xs  # Biomass yield coefficient (g/g)
        self.Sf = Sf  # Feed substrate concentration (g/L)
        self.F = F  # Feed flow rate (L/h)
        self.V0 = V0  # Initial reactor volume (L)
        self.S0 = S0  # Initial substrate concentration (g/L)
        self.C0 = C0  # Initial biomass concentration (g/L)
        self.P0 = P0  # Initial enzyme concentration (U/mL)
        self.alpha = alpha  # Growth-associated production coefficient (U/g)
        self.beta = beta  # Non-growth-associated production coefficient (U/g/h)
        self.t_max = t_max  # Simulation time (hours)

        self.n_x   = 4
        self.n_y   = 1
        self.n_u   = 1

        self.x0  = np.array([self.V0, self.C0, self.S0, self.P0])
    
    def f(self, x_step, u):

        V  = x_step[0:1]
        C  = x_step[1:2]
        S  = x_step[2:3]
        P  = x_step[3:4]
        F = u[0:1]
        
        # Model Dynamics
        dV = F
        if S > 0:
            mu = self.mu_max * (S / (self.Ks + S))
            dC = (mu - F/V) * C  # Biomass growth with dilution
            dS = (F/V) * (self.Sf - S) - (1/self.Y_xs) * dC  # Substrate balance
        else:
            dX = - (F/V) * C  # No growth if S = 0, just dilution
            dS = (F/V) * (self.Sf - S)  # Only feed addition if depleted

        dP = self.alpha * dC + self.beta * C - (F/V) * P  # Invertase production


        dx = np.concat([dV, dC, dS, dP])

        return dx
    
    def h(self, x_step):
        return np.array([x_step[3:4] * x_step[0:1]])
    
    def __call__(self, u, x0):

        if len(u.shape) == 1:
            print(f"u.shape: {u.shape} -> returning y0, x0 as the output")
            u = u.reshape(1, -1)

        T = u.shape[0]
        x = np.zeros((T, self.n_x))
        y = np.zeros((T, self.n_y))

        x[0, :] = x0
        y[0, :] = self.h(x_step=x0)
        for ind in range(T-1):
            y[ind + 1, :], x[ind + 1, :] = self.update(u=u[ind, :], x=x[ind, :])
            
        return y, x
    
    def update(self, u, x):
        x_updated = x + self.dt * self.f(x_step=x, u=u)
        y_updated = self.h(x_step=x_updated)
        return y_updated, x_updated