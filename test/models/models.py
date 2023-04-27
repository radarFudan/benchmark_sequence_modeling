import os
import pytest

import torch


from src.models.models import StableRNN


def test_stablernn():
    model = StableRNN(128)
    x = torch.randn(1, 100, 1)
    y = model(x)

    print(y.shape)

    print(y)
