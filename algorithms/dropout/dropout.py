import numpy as np
import torch
from torch import Tensor

from algorithms.basic_bo.bo import BO
from algorithms.dropout.unimportant_strategy import (
    RandomStrategy,
    BestKStrategy,
)
from algorithms.utils import latin_hypercube, from_unit_cube


def select_active_dim(dim, active_dim):
    idx = np.random.choice(range(dim), active_dim, replace=False)
    idx = np.sort(idx)
    return idx


class Dropout:
    def __init__(
        self,
        dim: int,
        lb: Tensor,
        ub: Tensor,
        active_dim: int,
        name: str = 'Dropout',
        n_init: int = 10,
        q: int = 1,
        inner_algo: str = 'BO',
        unimportant_strategy: str = 'bestk',
        k: int = 10,
        **inner_config,
    ):
        assert lb.ndim == 1 and ub.ndim == 1
        assert lb.shape == ub.shape
        assert (lb < ub).all()
        # assert active_dim <= dim
        assert inner_algo in ['BO']
        assert unimportant_strategy in ['random', 'bestk']
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.active_dim = min(active_dim, dim)
        self.name = name
        self.n_init = n_init
        self.q = q
        self.inner_algo = inner_algo
        self.k = k
        self.unimportant_strategy = self.create_unimportant_strategy(unimportant_strategy)
        self.inner_config = inner_config

        self.X = torch.zeros((0, dim))
        self.Y = torch.zeros((0, 1))

    def create_unimportant_strategy(self, strategy_name):
        if strategy_name == 'random':
            strategy = RandomStrategy(self.dim, self.lb, self.ub)
        elif strategy_name == 'bestk':
            strategy = BestKStrategy(self.dim, self.lb, self.ub, self.k)
        else:
            raise NotImplementedError
        return strategy

    def init(self):
        init_X = latin_hypercube(self.n_init, self.dim)
        init_X = from_unit_cube(init_X, self.lb.detach().cpu().numpy(), self.ub.detach().cpu().numpy())
        init_X = torch.from_numpy(init_X)
        return init_X

    def create_inner_algo(self, dim, lb, ub):
        if self.inner_algo == 'BO':
            algo = BO(dim, lb, ub, name='Dropout-BO', **self.inner_config)
        else:
            assert 0
        return algo

    def fill(self, idx, X):
        next_X = []
        for x in X:
            new_x = self.unimportant_strategy.fill(idx, x)
            next_X.append(new_x)
        next_X = torch.vstack(next_X)
        return next_X

    def ask(self):
        if len(self.X) == 0:
            next_X = self.init()
        else:
            idx = select_active_dim(self.dim, self.active_dim)

            # init the inner algorithm
            select_lb, select_ub = self.lb[idx], self.ub[idx]
            algo = self.create_inner_algo(self.active_dim, select_lb, select_ub)
            train_X = self.X[:, idx]
            train_Y = self.Y
            algo.tell(train_X, train_Y)

            # optimize
            important_X = algo.ask()

            # fill
            next_X = self.fill(idx, important_X)

        return next_X

    def tell(self, X: Tensor, Y: Tensor):
        self.X = torch.vstack((self.X, X))
        self.Y = torch.vstack((self.Y, Y))

        for x, y in zip(X, Y):
            self.unimportant_strategy.update(x, y)