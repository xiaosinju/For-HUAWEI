from typing import Optional
import logging

import torch 
from torch import Tensor, optim
import botorch
from botorch import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import Kernel, ScaleKernel, MaternKernel
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.constraints import GreaterThan

from algorithms.basic_bo.wrapper import (
    WrapperMean,
    WrapperKernel,
)
from algorithms.utils import latin_hypercube, from_unit_cube, timer_wrapper

log = logging.getLogger(__name__)


class BO:
    def __init__(
        self,
        dim: int,
        lb: Tensor,
        ub: Tensor,
        name: str = 'BO',
        n_init: int = 10,
        q: int = 1,
        mean: str = 'constant',
        kernel: str = 'matern',
        wrapper: str = 'identity',
        mll_opt: str = 'l-bfgs',
        acqf: str = 'EI',
        acqf_opt: str = 'l-bfgs',
        device: str = 'cpu',
        pretrain_file: str = None,
        finetune: bool = False,
        **kwargs,
    ):
        """
        Inputs:
            mll_opt_lr: float
            mll_opt_epochs: int
            hidden_dim_list: List
            kernel_out_features: int
        """
        assert lb.ndim == 1 and ub.ndim == 1
        assert lb.shape == ub.shape
        assert (lb < ub).all()
        assert mean in ['constant', 'mlp']
        assert kernel in ['rbf', 'matern']
        assert wrapper in ['identity', 'kumar', 'mlp']
        assert mll_opt in ['l-bfgs', 'adam']
        assert acqf in ['EI']
        assert acqf_opt in ['l-bfgs']
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.name = name
        self.n_init = n_init
        self.q = q
        self.mean = mean
        self.kernel = kernel
        self.wrapper = wrapper
        self.mll_opt = mll_opt
        self.acqf = acqf
        self.acqf_opt = acqf_opt
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.pretrain_file = pretrain_file
        self.finetune = finetune

        # parameters for mll opt
        self.mll_opt_lr = 0.01
        self.mll_opt_epochs = 300

        # parameters for mlp kernel
        self.hidden_dim_list = kwargs.get('hidden_dim_list', [])
        self.kernel_out_features = kwargs.get('kernel_out_features', dim)

        self.X = torch.zeros((0, dim))
        self.Y = torch.zeros((0, 1))

    def init(self):
        init_X = latin_hypercube(self.n_init, self.dim)
        init_X = from_unit_cube(init_X, self.lb.detach().cpu().numpy(), self.ub.detach().cpu().numpy())
        init_X = torch.from_numpy(init_X)
        return init_X

    def create_model(self, train_X, train_Y):
        if self.mean == 'constant':
            mean_module = ConstantMean()
        elif self.mean == 'mlp':
            mean_module = WrapperMean(self.dim, hidden_dim_list=self.hidden_dim_list)
        else:
            raise NotImplementedError
        dim = self.dim if self.wrapper != 'mlp' else self.kernel_out_features
        base_kernel = ScaleKernel(MaternKernel(ard_num_dims=dim))
        # base_kernel = ScaleKernel(
        #     MaternKernel(
        #         nu=2.5,
        #         ard_num_dims=train_X.shape[-1],
        #         lengthscale_prior=GammaPrior(3.0, 6.0),
        #     ),
        #     outputscale_prior=GammaPrior(2.0, 0.15),
        # )
        covar_module = WrapperKernel(
            base_kernel,
            self.dim,
            out_features=self.kernel_out_features,
            wrapper=self.wrapper,
            hidden_dim_list=self.hidden_dim_list,
        )
        model = SingleTaskGP(train_X, train_Y, covar_module=covar_module, mean_module=mean_module)
        model.likelihood.noise_covar.register_constraint('raw_noise', GreaterThan(1e-4))
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

        if self.pretrain_file is not None:
            path = 'saved_models/{}.pth'.format(self.pretrain_file)
            state_dict = torch.load(path)
            model.load_state_dict(state_dict, strict=False)
            log.info('Load from {}'.format(path))

        return mll, model

    def optimize_model(self, mll, model, train_X, train_Y):
        if self.pretrain_file is not None and not self.finetune:
            return 

        if self.mll_opt == 'l-bfgs':
            fit_gpytorch_mll(mll)
        elif self.mll_opt == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=self.mll_opt_lr)
            model.train()
            model.likelihood.train()
            for _ in range(self.mll_opt_epochs):
                optimizer.zero_grad()
                output = model(train_X)
                loss = - mll(output, train_Y.reshape(-1))
                loss.backward()
                optimizer.step()
            model.eval()
            model.likelihood.eval()
        else:
            raise NotImplementedError

    def create_acqf(self, model, train_X, train_Y):
        if self.acqf == 'EI':
            AF = ExpectedImprovement(model, train_Y.max())
        else:
            raise NotImplementedError
        return AF

    def optimize_acqf(self, AF):
        bounds = torch.vstack((torch.zeros(self.dim), torch.ones(self.dim))).double().to(self.device)
        if self.acqf_opt == 'l-bfgs':
            next_X, _ = botorch.optim.optimize.optimize_acqf(AF, bounds=bounds, q=self.q, num_restarts=10, raw_samples=1024)
        else:
            raise NotImplementedError

        assert next_X.shape == (self.q, self.dim)
        return next_X

    def preprocess(self):
        train_X = (self.X - self.lb) / (self.ub - self.lb)
        train_Y = (self.Y - self.Y.mean()) / (self.Y.std() + 1e-6)
        train_X, train_Y = train_X.to(self.device), train_Y.to(self.device)
        
        return train_X.double(), train_Y.double()

    def postprocess(self, next_X):
        next_X = next_X.to('cpu')
        next_X = self.lb + next_X * (self.ub - self.lb)
        return next_X

    def ask(self) -> Tensor:
        """
        Outputs:
            Tensor with shape (q, dim)
        """
        if len(self.X) == 0:
            next_X = self.init()
        else:
            train_X, train_Y = self.preprocess()
            mll, model = self.create_model(train_X, train_Y)
            self.optimize_model(mll, model, train_X, train_Y)
            AF = self.create_acqf(model, train_X, train_Y)
            next_X = self.optimize_acqf(AF)
            next_X = self.postprocess(next_X)

        return next_X

    def tell(self, X: Tensor, Y: Tensor) -> Tensor:
        """
        Inputs:
            X: Tensor with shape (bs, dim)
            Y: Tensor with shape (bs, 1)
        """
        X, Y = X.to(self.X), Y.to(self.Y)
        self.X = torch.vstack((self.X, X))
        self.Y = torch.vstack((self.Y, Y))