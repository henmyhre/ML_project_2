import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryClassfier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BinaryClassfier, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x1 = F.leaky_relu(self.linear_1(x))
        return self.linear_2(x1)
