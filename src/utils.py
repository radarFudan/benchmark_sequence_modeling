import numpy as np

import argparse

import torch
import torch.nn as nn


def reset_torch():
    np.random.seed(123)
    torch.manual_seed(123)
    torch.use_deterministic_algorithms(True)
