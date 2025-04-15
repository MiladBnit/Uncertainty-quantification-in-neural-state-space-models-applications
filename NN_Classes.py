import torch
import torch.nn as nn
from typing import List

####################################################


def digits_repr(num, base, min_digits):
    res = []
    while num:
        res.append(num % base)
        num //= base

    if len(res) < min_digits:
        res = res + [0] * (min_digits - len(res))
    res.reverse()
    return res


def valid_coeffs(n_feat, p):
    n_comb = (p+1)**n_feat  # combination of possible monomials
    pows = []
    for comb in range(1, n_comb+1):
        pow = digits_repr(comb, base=p+1, min_digits=n_feat)
        if 1 < sum(pow) < p+1:
            pows.append(pow)
    return pows


# Classes from torchid repo.
class StateSpaceSimulator(nn.Module):
    r""" Discrete-time state-space simulator.

    Args:
        f_xu (nn.Module): The neural state-space model.
        batch_first (bool): If True, first dimension is batch.

    Inputs: x_0, u
        * **x_0**: tensor of shape :math:`(N, n_{x})` containing the
          initial hidden state for each element in the batch.
          Defaults to zeros if (h_0, c_0) is not provided.
        * **input**: tensor of shape :math:`(L, N, n_{u})` when ``batch_first=False`` or
          :math:`(N, L, n_{x})` when ``batch_first=True`` containing the input sequence

    Outputs: x
        * **x**: tensor of shape :math:`(L, N, n_{x})` corresponding to
          the simulated state sequence.

    Examples::

        >>> ss_model = NeuralStateSpaceModel(n_x=3, n_u=2)
        >>> nn_solution = StateSpaceSimulator(ss_model)
        >>> x0 = torch.randn(64, 3)
        >>> u = torch.randn(100, 64, 2)
        >>> x = nn_solution(x0, u)
        >>> print(x.size())
        torch.Size([100, 64, 3])
     """

    def __init__(self, f_xu, g_x=None, batch_first=False):
        super().__init__()
        self.f_xu = f_xu
        self.g_x = g_x
        self.batch_first = batch_first

    def simulate_state(self, x_0, u):
        x: List[torch.Tensor] = []
        x_step = x_0
        dim_time = 1 if self.batch_first else 0

        for u_step in u.split(1, dim=dim_time):  # split along the time axis
            u_step = u_step.squeeze(dim_time)
            x += [x_step]
            dx = self.f_xu(x_step, u_step)
            x_step = x_step + dx

        x = torch.stack(x, dim_time)
        return x

    def forward(self, x_0, u, return_x=False):
        x = self.simulate_state(x_0, u)
        if self.g_x is not None:
            y = self.g_x(x)
        else:
            y = x
        if not return_x:
            return y
        else:
            return y, x

####################################################

class NeuralStateUpdate(nn.Module):
    r"""State-update mapping modeled as a feed-forward neural network with one hidden layer.

    The model has structure:

    .. math::
        \begin{aligned}
            x_{k+1} = x_k + \mathcal{N}(x_k, u_k),
        \end{aligned}

    where :math:`\mathcal{N}(\cdot, \cdot)` is a feed-forward neural network with one hidden layer.

    Args:
        n_x (int): Number of state variables
        n_u (int): Number of input variables
        hidden_size: (int, optional): Number of input features in the hidden layer. Default: 0
        init_small: (boolean, optional): If True, initialize to a Gaussian with mean 0 and std 10^-4. Default: True
        activation: (str): Activation function in the hidden layer. Either 'relu', 'softplus', 'tanh'. Default: 'relu'

    Examples::

        >>> ss_model = NeuralStateUpdate(n_x=2, n_u=1, hidden_size=64)
    """

    def __init__(self, n_x, n_u, hidden_size=16, init_small=True):
        super(NeuralStateUpdate, self).__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.n_feat = hidden_size
        self.net = nn.Sequential(
            nn.Linear(n_x + n_u, hidden_size),  # 2 states, 1 input
            nn.Tanh(),
            nn.Linear(hidden_size, n_x)
        )

        if init_small:
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, x, u):
        xu = torch.cat((x, u), -1)
        dx = self.net(xu)
        return dx


class PolynomialStateUpdate(nn.Module):
    r"""State-update mapping modeled as a polynomial in x and u.

    The model has structure:

    .. math::
        \begin{aligned}
            x_{k+1} = x_k + Ax_{k} + Bu_{k} + Ez_{k},
        \end{aligned}

    where z_{k} is a vector containing (non-linear) monomials in x_{k} and u_{k}

    Args:
        n_x: (np.array): Number of states.
        n_u: (np.array): Number of inputs.
        d_max (int): Maximum degree of the polynomial model.

    """

    def __init__(self, n_x, n_u, d_max=1, init_small=True, hidden_size=None):
        super(PolynomialStateUpdate, self).__init__()

        self.n_x = n_x
        self.n_u = n_u
        poly_coeffs = valid_coeffs(n_x + n_u, d_max)
        self.n_poly = len(poly_coeffs)
        self.poly_coeffs = torch.tensor(poly_coeffs)
        self.A = nn.Linear(n_x, n_x, bias=False)
        self.B = nn.Linear(n_u, n_x, bias=False)
        # self.D = nn.linear(n_u, n_y, bias=False)
        self.E = nn.Linear(self.n_poly, n_x, bias=False)
        # self.F = nn.linear(self.n_poly, n_y)
        self.nl_on = True

        if init_small:
            nn.init.normal_(self.A.weight, mean=0, std=1e-3)
            nn.init.normal_(self.B.weight, mean=0, std=1e-3)
            nn.init.normal_(self.E.weight, mean=0, std=1e-6)

            # nn.init.constant_(module.bias, val=0)

    def enable_nl(self):
        self.nl_on = True

    def disable_nl(self):
        self.nl_on = False

    def freeze_nl(self):
        self.E.requires_grad_(False)

    def unfreeze_nl(self):
        self.E.requires_grad_(True)

    def freeze_lin(self):
        self.A.requires_grad_(False)
        self.B.requires_grad_(False)

    def unfreeze_lin(self):
        self.A.requires_grad_(True)
        self.B.requires_grad_(True)

    def forward(self, x, u):
        xu = torch.cat((x, u), dim=-1)
        xu_ = xu.unsqueeze(xu.ndim - 1)

        dx = self.A(x) + self.B(u)
        if self.nl_on:
            zeta = torch.prod(torch.pow(xu_, self.poly_coeffs), axis=-1)
            # eta = torch.prod(torch.pow(xu_, self.poly_coeffs), axis=-1)
            dx = dx + self.E(zeta)
        return dx


class NeuralLinStateUpdate(nn.Module):
    r"""State-update mapping modeled as a feed-forward neural network with one hidden layer.

    The model has structure:

    .. math::
        \begin{aligned}
            x_{k+1} = x_k + \mathcal{N}(x_k, u_k),
        \end{aligned}

    where :math:`\mathcal{N}(\cdot, \cdot)` is a feed-forward neural network with one hidden layer.

    Args:
        n_x (int): Number of state variables
        n_u (int): Number of input variables
        hidden_size: (int, optional): Number of input features in the hidden layer. Default: 0
        init_small: (boolean, optional): If True, initialize to a Gaussian with mean 0 and std 10^-4. Default: True
        activation: (str): Activation function in the hidden layer. Either 'relu', 'softplus', 'tanh'. Default: 'relu'

    Examples::

        >>> ss_model = NeuralStateUpdate(n_x=2, n_u=1, hidden_size=64)
    """

    def __init__(self, n_x, n_u, hidden_size=16, init_small=True):
        super(NeuralLinStateUpdate, self).__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(n_x + n_u, hidden_size),  # 2 states, 1 input
            nn.Tanh(),
            nn.Linear(hidden_size, n_x)
        )
        self.lin = nn.Linear(n_x + n_u, n_x, bias=False)
        self.nl_on = True

        if init_small:
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

            for m in self.lin.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)

    def freeze_nl(self):
        self.net.requires_grad_(False)

    def unfreeze_nl(self):
        self.net.requires_grad_(True)

    def freeze_lin(self):
        self.lin.requires_grad_(False)

    def unfreeze_lin(self):
        self.lin.requires_grad_(True)

    def enable_nl(self):
        self.nl_on = True

    def disable_nl(self):
        self.nl_on = False

    def forward(self, x, u):
        xu = torch.cat((x, u), -1)
        dx = self.lin(xu)
        if self.nl_on:
            dx = dx + self.net(xu)
        return dx


class LinearStateUpdate(nn.Module):
    r"""State-update mapping modeled as a linear function in x and u.

    The model has structure:

    .. math::
        \begin{aligned}
            x_{k+1} = x_k + Ax_{k} + Bu_{k}.
        \end{aligned}

    Args:
        n_x: (np.array): Number of states.
        n_u: (np.array): Number of inputs.
        d_max (int): Maximum degree of the polynomial model.

    """

    def __init__(self, n_x, n_u, init_small=True, hidden_size=None):
        super(LinearStateUpdate, self).__init__()

        self.n_x = n_x
        self.n_u = n_u
        self.A = nn.Linear(n_x, n_x, bias=False)
        self.B = nn.Linear(n_u, n_x, bias=False)

        if init_small:
            for module in [self.A, self.B]:
                nn.init.normal_(module.weight, mean=0, std=1e-2)
                # nn.init.constant_(module.bias, val=0)

    def forward(self, x, u):
        dx = self.A(x) + self.B(u)
        return dx


class CTSNeuralStateSpace(nn.Module):
    r"""A state-space model to represent the cascaded two-tank system.

    Args:
        hidden_size: (int, optional): Number of input features in the hidden layer. Default: 0
        init_small: (boolean, optional): If True, initialize to a Gaussian with mean 0 and std 10^-4. Default: True
    """

    def __init__(self, n_x, n_u, hidden_size=64, init_small=True):
        super(CTSNeuralStateSpace, self).__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.n_feat = hidden_size
        self.net = nn.Sequential(
            nn.Linear(n_x + n_u, hidden_size),  # 2 states, 1 input
            nn.ReLU(),
            nn.Linear(hidden_size, n_x)
        )

        if init_small:
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, x, u):
        xu = torch.cat((x, u), -1)
        dx = self.net(xu)
        return dx


class LinearOutput(nn.Module):
    r"""Output  mapping modeled as a linear function in x.

    The model has structure:

    .. math::
        \begin{aligned}
            y_{k} = Cx_k.
        \end{aligned}
    """

    def __init__(self, n_x, n_y, bias=False, hidden_size=None):
        super(LinearOutput, self).__init__()
        self.n_x = n_x
        self.n_y = n_y
        self.C = torch.nn.Linear(n_x, n_y, bias=bias)

    def forward(self, x):
        return self.C(x)


class NeuralOutput(nn.Module):
    r"""Output  mapping modeled as a feed-forward neural network in x.

    The model has structure:

    .. math::
        \begin{aligned}
            y_{k} = \mathcal{N}(x_k).
        \end{aligned}
    """

    def __init__(self, n_x, n_y, hidden_size=16):
        super(NeuralOutput, self).__init__()
        self.n_x = n_x
        self.n_y = n_y
        self.net = nn.Sequential(nn.Linear(n_x, hidden_size),
                                 nn.Tanh(),
                                 nn.Linear(hidden_size, n_y)
                                 )

    def forward(self, x):
        return self.net(x)


class NeuralLinOutput(nn.Module):
    r"""Output  mapping modeled as a feed-forward neural network in x.

    The model has structure:

    .. math::
        \begin{aligned}
            y_{k} = \mathcal{N}(x_k).
        \end{aligned}
    """

    def __init__(self, n_x, n_y, hidden_size=16):
        super(NeuralLinOutput, self).__init__()
        self.n_x = n_x
        self.n_y = n_y
        self.net = nn.Sequential(nn.Linear(n_x, hidden_size),
                                 nn.Tanh(),
                                 nn.Linear(hidden_size, n_y)
                                 )

        self.lin = nn.Linear(n_x, n_y, bias=False)
        self.nl_on = True

    def freeze_nl(self):
        self.net.requires_grad_(False)

    def unfreeze_nl(self):
        self.net.requires_grad_(True)

    def freeze_lin(self):
        self.lin.requires_grad_(False)

    def unfreeze_lin(self):
        self.lin.requires_grad_(True)

    def enable_nl(self):
        self.nl_on = True

    def disable_nl(self):
        self.nl_on = False

    def forward(self, x):
        y = self.lin(x)
        if self.nl_on:
            y += self.net(x)
        return y


class ChannelsOutput(nn.Module):
    r"""Output  mapping corresponding to a specific state channel.

    """

    def __init__(self, channels):
        super(ChannelsOutput, self).__init__()
        self.channels = channels

    def forward(self, x):
        y = x[..., self.channels]
        return y


class f_ss_class(nn.Module):
    def __init__(self, n_x, n_u, init_small=True, hidden_size=None):
        super(f_ss_class, self).__init__()
        self.n_x = n_x
        self.n_u = n_u
        A = torch.zeros((n_x, n_x), dtype=torch.float32, requires_grad=True)
        B = torch.zeros((n_x, n_u), dtype=torch.float32, requires_grad=True)
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)

    def forward(self, x, u):
        if x.ndim > 1:
            T = x.shape[0]
            dx = torch.zeros((T, self.n_x))
            for k in range(T):
                dx[k, :] = self.A @ x[k, :] + self.B @ u[k, :]
            return dx
        else:
            x_step = x
            u_step = u
            dx_step = self.A @ x_step + self.B @ u_step
            return dx_step
        

class f_ss_diag_class(nn.Module):
    def __init__(self, n_x, n_u, init_small=True, hidden_size=None):
        super(f_ss_diag_class, self).__init__()
        self.n_x = n_x
        self.n_u = n_u
        A_diag = torch.zeros(n_x, dtype=torch.float32, requires_grad=True)
        B = torch.zeros((n_x, n_u), dtype=torch.float32, requires_grad=True)
        self.A_diag = nn.Parameter(A_diag)
        self.B = nn.Parameter(B)

    def forward(self, x, u):
        A = torch.diag(self.A_diag)
        if x.ndim > 1:
            T = x.shape[0]
            dx = torch.zeros((T, self.n_x))
            for k in range(T):
                dx[k, :] = A @ x[k, :] + self.B @ u[k, :]
            return dx
        else:
            x_step = x
            u_step = u
            dx_step = A @ x_step + self.B @ u_step
            return dx_step
        

class g_ss_class(nn.Module):
    def __init__(self, n_x, n_y, hidden_size=None):
        super(g_ss_class, self).__init__()
        self.n_x = n_x
        self.n_y = n_y
        C = torch.zeros((n_y, n_x), dtype=torch.float32)
        C[0, 0] = 1
        self.C = nn.Parameter(C)

    def forward(self, x):
        if x.ndim > 1:
            T = x.shape[0]
            y = torch.zeros((T, self.n_y))
            for k in range(T):
                y[k, :] = self.C @ x[k, :]
            return y
        else:
            x_step = x
            y_step = self.C @ x_step
            return y_step