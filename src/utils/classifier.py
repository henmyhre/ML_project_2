import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryClassfier_two_layer(nn.Module):
    """
    Initializes a neural network with two hidden layers.
    """
    def __init__(self, input_size, hidden_size_1=100, hidden_size_2=20):
        super(BinaryClassfier_two_layer, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size_1)
        self.linear_2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear_3 = nn.Linear(hidden_size_2, 1)
                
    def forward(self, x):
        x1 = F.leaky_relu(self.linear_1(x))
        x2 = F.leaky_relu(self.linear_2(x1))
        return self.linear_3(x2)


class BinaryClassfier_one_layer(nn.Module):
    """
    Initializes a neural network with one hidden layers.
    """
    def __init__(self, input_size, hidden_size_1=100):
        super(BinaryClassfier_one_layer, self).__init__()
        self.linear_1 = nn.Linear(input_size, hidden_size_1)
        self.linear_2 = nn.Linear(hidden_size_1, 1)
       
    def forward(self, x):
        x1 = F.leaky_relu(self.linear_1(x))
        return self.linear_2(x1)
  
    
      
class LogisticRegression(torch.nn.Module):
    """
    Initializes a logistic regression model.
    """
    def __init__(self, input_size, output_size = 1):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs
      