import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class MLP(nn.Module):
  def __init__(self, hidden_size):
    super(MLP, self).__init__()

    self.linear_1 = nn.Linear(1, hidden_size)
    self.linear_2 = nn.Linear(hidden_size, hidden_size)
    self.linear_3 = nn.Linear(hidden_size, 1)
  
  def forward(self, x):
    x1 = torch.sigmoid(self.linear_1(x))
    x2 = torch.sigmoid(self.linear_2(x1))
    return self.linear_3(x2)
    
def train(features, labels, model, lossfunc, optimizer, num_epoch, BATCH_SIZE):
  
  """train a model for num_epoch epochs on the given data
  
  Inputs:
      features: a numpy array
      labels: a numpy array
      model: an instance of nn.Module (or classes with similar signature)
      lossfunc: a function : (prediction outputs, correct outputs) -> loss
      optimizer: an instance of torch.optim.Optimizer
      num_epoch: an int
      BATCH_SIZE: an int
  """
  
  features_torch = torch.from_numpy(features)
  labels_torch = torch.from_numpy(labels)
  dataset = torch.utils.data.TensorDataset(features_torch, labels_torch)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
  
  for epoch in range(num_epoch):
    
    for id_batch, (x_batch, y_batch) in enumerate(dataloader):
      
      pred = model.forward(x_batch)
      loss = lossfunc(pred, y_batch)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    
    if epoch % 10 == 0:
      print ('Epoch [%d/%d], Loss: %.4f' %(epoch+1, num_epoch, loss.item()))
      