import scipy

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P


class PositiveDefinite(nn.Module):
    def forward(self, X):
        return torch.matmul(X, X.T)  # Return a positive definite matrix

    def right_inverse(self, A):
        return torch.linalg.pinv(A)


class ExpLambda(nn.Module):
    """A parameterization to make the weight matrix positive definite"""

    def forward(self, X):
        return torch.matrix_exp(X)  # Return an exponential parametrized matrix

    def right_inverse(self, A):
        return torch.real(torch.from_numpy(scipy.linalg.logm(A)))
