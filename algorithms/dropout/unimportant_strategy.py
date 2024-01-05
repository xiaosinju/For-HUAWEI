from abc import ABCMeta, abstractmethod

import torch
from torch import Tensor


class BaseStrategy:
    def __init__(self, dim, lb: Tensor, ub: Tensor):
        """
        Inputs:
            dim: int, the total number of variables
            lb: Tensor with shape (dim, )
            ub: Tensor with shape (dim, )
        """
        assert lb.ndim == 1 and ub.ndim == 1
        self.dim = dim
        self.lb = lb
        self.ub = ub

    @abstractmethod
    def fill(self, important_idx: list, important_x: Tensor):
        """
        Inputs:
        """

    @abstractmethod
    def update(self, x: Tensor, y: Tensor):
        """
        Inputs:
            x: Tensor with shape (dim, )
            y: Tensor with shape (1, )
        """
   

class RandomStrategy(BaseStrategy):
    def fill(self, important_idx, important_x):
        idx2x = dict(zip(important_idx, important_x))
        new_x = torch.zeros(self.dim)
        for i in range(self.dim):
            if i in important_idx:
                new_x[i] = idx2x[i]
            else:
                new_x[i] = self.lb[i] + (self.ub[i] - self.lb[i]) * torch.rand(1)
        return new_x

    def update(self, x, y):
        pass

    
class BestKStrategy(BaseStrategy):
    def __init__(self, dim, lb, ub, k=10):
        super(BestKStrategy, self).__init__(dim, lb, ub)
        self.k = k
        self.best_X = torch.zeros((0, dim))
        self.best_Y = torch.zeros((0, 1))

    def fill(self, important_idx, important_x):
        idx2x = dict(zip(important_idx, important_x))
        new_x = torch.zeros(self.dim)
        for i in range(self.dim):
            if i in important_idx:
                new_x[i] = idx2x[i]
            else:
                fill_idx = torch.randint(low=0, high=len(self.best_X), size=(1, )).item()
                new_x[i] = self.best_X[fill_idx][i]
        return new_x

    def update(self, x, y):
        if len(self.best_X) < self.k:
            self.best_X = torch.vstack((self.best_X, x))
            self.best_Y = torch.vstack((self.best_Y, y))
        else:
            worst_y = self.best_Y.min()
            if y > worst_y:
                worst_choice = torch.argwhere(torch.isclose(self.best_Y, worst_y)).reshape(-1)
                idx = torch.randint(low=0, high=len(worst_choice), size=(1, )).item()
                worst_idx = worst_choice[idx].item()
                self.best_X[worst_idx] = x
                self.best_Y[worst_idx] = y
        assert len(self.best_X) <= self.k