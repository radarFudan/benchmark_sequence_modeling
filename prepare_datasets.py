import os
from pathlib import Path

import numpy as np

from sklearn.model_selection import train_test_split

from data.lfnlf import *


# In general, for sequence dataset, the purpose is to learn the relationship between
# (N, T, I) to (N, T, O)

# LF, NLF
task = "LF"
seq_len = 64
N = 100000
memory_type = "exp"

input_dim = 1
output_dim = 1
rho = np.zeros((seq_len, input_dim, output_dim))
for i in range(seq_len):
    rho[i, :, :] = np.exp(-i)

save_dir = Path(f"./data/{task}_{N}/{memory_type}")
save_dir.mkdir(exist_ok=True, parents=True)
x, y = lfnlf(N, seq_len, input_dim, output_dim, rho, power=1.0 if task == "LF" else 2.0)

# to do the train, test split
x_train, y_train, x_valid, y_valid = train_test_split(
    x, y, test_size=0.2, random_state=2023
)

print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)
torch.save(torch.Tensor(x_train), save_dir / "x_train.pt")
torch.save(torch.Tensor(y_train), save_dir / "y_train.pt")
torch.save(torch.Tensor(x_valid), save_dir / "x_valid.pt")
torch.save(torch.Tensor(y_valid), save_dir / "y_valid.pt")
