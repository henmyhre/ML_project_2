import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class MLP(nn.Module):
  def __init__(self, hidden_size, lossfunc):
    super(MLP, self).__init__()

    self.linear_1 = nn.Linear(1, hidden_size)
    self.linear_2 = nn.Linear(hidden_size, hidden_size)
    self.linear_3 = nn.Linear(hidden_size, 1)

    self.lossfunc = lossfunc

  def set_optimizer(self):
    self.optimizer=torch.optim.SGD(self.parameters(), lr=1e-3)
    
  def forward(self, x):
    x1 = torch.sigmoid(self.linear_1(x))
    x2 = torch.sigmoid(self.linear_2(x1))
    return self.linear_3(x2)
    
  def train(self, features, labels, num_epoch, BATCH_SIZE):
    
    """
    Train a model for num_epoch epochs on the given data
    
    Inputs:
      self: an instance of nn.Module (or classes with similar signature)
      features: a numpy array
      labels: a numpy array
      num_epoch: an int
      BATCH_SIZE: an int
    """
    
    features_torch = torch.from_numpy(features)
    labels_torch = torch.from_numpy(labels)
    dataset = torch.utils.data.TensorDataset(features_torch, labels_torch)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(num_epoch):
      
      for id_batch, (x_batch, y_batch) in enumerate(dataloader):
        
        pred = self.forward(x_batch)
        loss = self.lossfunc(pred, y_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
      
      if epoch % 10 == 0:
        print ('Epoch [%d/%d], Loss: %.4f' %(epoch+1, num_epoch, loss.item()))
        