import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, in_dim=768 + 256, hidden_dim=32):
        super(NN, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

    def embed(self, x):
        return F.relu(self.linear1(x))
