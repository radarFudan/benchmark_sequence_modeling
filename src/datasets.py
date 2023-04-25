import numpy as np
import pandas as pd

import torch

import einops

# Synthesized dataset


def lfnlf(
    N,
    seq_len,
    input_dim=1,
    output_dim=1,
    rho=None,
    make_input_gaussian_process=False,
    power=1.0,
):
    """
    Generate LF-NLF dataset

    Notice that rho is a 3D tensor, with shape (seq_len, input_dim, output_dim)
    And rho(seq_len - 1) is the largest memory.
    """
    if rho is None:
        np.random.randn(N, input_dim, output_dim)

    inputs = np.random.normal(size=(N, seq_len, input_dim))
    outputs = np.zeros((N, seq_len, output_dim))

    if make_input_gaussian_process:
        inputs = np.cumsum(inputs, axis=1)

    if power != 1.0:
        powered_inputs = inputs * (np.power(np.abs(inputs), power - 1))
    else:
        powered_inputs = inputs

    for i in range(seq_len):
        outputs[:, i, :] = einops.einsum(
            powered_inputs[:, : i + 1, :],
            rho[seq_len - 1 - i :, :, :],
            "N T i, T i o -> N o",
        )
    return inputs, outputs


if __name__ == "__main__":
    N = 2
    seq_len = 3
    input_dim = 1
    output_dim = 1
    rho = np.ones((seq_len, input_dim, output_dim))
    inputs, outputs = lfnlf(N, seq_len, input_dim, output_dim, rho)
    print(inputs.shape, outputs.shape)

    print(inputs)
    print(outputs)
