import torch.nn as nn


class QDS(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(600, 300)
        self.linear2 = nn.Linear(300, 100)
        self.linear3 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear3((self.linear2(self.linear(x)))))
