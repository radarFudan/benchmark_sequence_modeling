import sys
import time
import shutil
from pathlib import Path

import torch

task = "LF"
seq_len = 64
N = 100000
memory_type = "exp"

input_dim = 1
output_dim = 1
# rho = None

data_dir = Path(f"./data/{task}_{N}/{memory_type}")
if not data_dir.exists():
    print(f"Data directory {data_dir} does not exist.")
    sys.exit(1)

# Model construction
# Vanilla RNN
model = torch.nn.RNN(
    input_size=input_dim,
    hidden_size=64,
    num_layers=1,
    nonlinearity="tanh",
    bias=True,
    batch_first=True,
    dropout=0.0,
    bidirectional=False,
)

# Model training
# Hyperparameters
batch_size = 128
num_epochs = 100
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

# Load data
x_train = torch.load(data_dir / "x_train.pt")
y_train = torch.load(data_dir / "y_train.pt")
x_valid = torch.load(data_dir / "x_valid.pt")
y_valid = torch.load(data_dir / "y_valid.pt")

# Training
start_time = time.time()
for epoch in range(num_epochs):
    # Training
    model.train()
    for i in range(0, x_train.shape[0], batch_size):
        x_batch = x_train[i : i + batch_size, :, :]
        y_batch = y_train[i : i + batch_size, :, :]
        optimizer.zero_grad()
        y_pred = model(x_batch)[0]
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        y_pred = model(x_valid)[0]
        loss = criterion(y_pred, y_valid)
        print(f"Epoch {epoch}: {loss:.4f}")
