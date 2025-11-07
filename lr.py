import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # maps input_dim -> 1 output

    def forward(self, x):
        return self.linear(x)
