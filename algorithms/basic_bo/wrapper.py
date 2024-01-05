import torch 
from torch import nn 
from gpytorch.means import Mean
from gpytorch.kernels import Kernel


class Squareplus(nn.Module):
    def __init__(self):
        super(Squareplus, self).__init__()

    def forward(self, x):
        return 0.5 * (x + torch.sqrt(x**2 + 4))


class IdentityWrapper(nn.Module):
    def __init__(self):
        super(IdentityWrapper, self).__init__()

    def forward(self, x):
        return x


class MLPWrapper(nn.Module):
    def __init__(self, in_features, hidden_dim_list, out_features):
        super(MLPWrapper, self).__init__()
        self.mlp = []
        for i in hidden_dim_list:
            self.mlp.append(nn.Linear(in_features, i))
            self.mlp.append(nn.Tanh())
            in_features = i
        self.mlp.append(nn.Linear(in_features, out_features))
        self.mlp.append(nn.Tanh())
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x):
        return self.mlp(x)


class KumarWrapper(nn.Module):
    def __init__(self):
        super(KumarWrapper, self).__init__()
        self.transform = nn.Softplus()
        # self.transform = Squareplus()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.eps = 1e-6

    def forward(self, x):
        x = x.clip(self.eps, 1-self.eps)
        alpha = self.transform(self.alpha)
        beta = self.transform(self.beta)

        res = 1 - (1 - x.pow(alpha)).pow(beta)
        return res


def create_wrapper(wrapper, **kwargs):
    if wrapper == 'identity':
        wrapper = IdentityWrapper()
    elif wrapper == 'kumar':
        wrapper = KumarWrapper()
    elif wrapper == 'mlp':
        wrapper = MLPWrapper(
            kwargs['in_features'],
            kwargs['hidden_dim_list'],
            kwargs['out_features'],
        )
    else:
        raise NotImplementedError
    return wrapper


class WrapperMean(Mean):
    def __init__(self, dim, wrapper='mlp', hidden_dim_list=[8]):
        assert wrapper in ['mlp']
        assert len(hidden_dim_list) >= 1
        super(Mean, self).__init__()
        self.wrapper = create_wrapper(wrapper, in_features=dim, hidden_dim_list=hidden_dim_list[: -1], out_features=hidden_dim_list[-1])
        self.final_layer = nn.Linear(in_features=hidden_dim_list[-1], out_features=1)

    def forward(self, x):
        m = self.final_layer(self.wrapper(x))
        return m.squeeze()


class WrapperKernel(Kernel):
    def __init__(self, base_kernel, dim, out_features=None, wrapper='identity', hidden_dim_list=[8], **kwargs):
        assert wrapper != 'mlp' or out_features is not None
        assert wrapper != 'mlp' or len(hidden_dim_list) >= 1
        super(WrapperKernel, self).__init__(**kwargs)
        self.base_kernel = base_kernel
        self.wrapper = create_wrapper(
            wrapper,
            in_features=dim,
            hidden_dim_list=hidden_dim_list,
            out_features=out_features,
        )

    def forward(self, x1, x2, **params):
        x1 = self.wrapper(x1)
        x2 = self.wrapper(x2)
        return self.base_kernel(x1, x2, **params)