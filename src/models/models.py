import scipy

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P

from src.models.layers import ExpLambda


class StableRNN(nn.Module):
    def __init__(
        self,
        hid_dim=128,
        input_dim=1,
        output_dim=1,
        activation="linear",
        dt=0.1,
        dtype=torch.float64,
        bias=False,
    ):
        """_summary_

        Args:
            activation (_type_): _description_
            recurrent_initializer (_type_): _description_
            hid_dim (_type_): _description_
            input_dim (_type_): _description_
            output_dim (_type_): _description_

            reduced_dim (_type_): This reduced dimension is usually smalelr than the hidden dimension. Defaults to None.

            sum_terms (int, optional): _description_. Defaults to 40.
            dt (float, optional): _description_. Defaults to 0.1.
            beta (float, optional): _description_. Defaults to 0.1.
        """

        super().__init__()
        if activation == "linear":
            self.activation = None
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "hardtanh":
            self.activation = torch.nn.functional.hardtanh

        self.U = nn.Linear(input_dim, hid_dim, bias=bias, dtype=dtype)
        self.W = nn.Linear(hid_dim, hid_dim, bias=bias, dtype=dtype)
        self.c = nn.Linear(hid_dim, output_dim, bias=bias, dtype=dtype)

        # P.register_parametrization(self.W, "weight", PositiveDefinite()) # W = A A^T
        P.register_parametrization(self.W, "weight", ExpLambda())  # W = exp(A)

        self.dt = dt
        self.hid_dim = hid_dim

    def forward(self, x):
        # x = [batch size, input len, input dim]
        length = x.shape[1]
        x = self.U(x)

        hidden = []
        hidden.append(torch.zeros(1, 1, self.hid_dim, dtype=x.dtype, device=x.device))

        # Residual RNN
        for i in range(length):
            h_next = hidden[i] + self.dt * self.activation(
                x[:, i : i + 1, :] - self.W(hidden[i])
            )
            hidden.append(h_next)
        hidden = torch.cat(hidden[1:], dim=1)

        out = self.c(hidden)
        # y = [batch size, input len, output dim]
        return out
