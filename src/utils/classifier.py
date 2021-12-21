import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class BinaryClassfier(nn.Module):
    def __init__(self, device, input_size, hidden_size):
        super(BinaryClassfier, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size, device=device)
        self.linear_2 = nn.Linear(hidden_size, 1, device=device)
        
    def forward(self, x):
        x1 = F.leaky_relu(self.linear_1(x))
        return torch.sigmoid(self.linear_2(x1))
