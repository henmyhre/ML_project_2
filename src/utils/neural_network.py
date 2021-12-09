import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class MLP(nn.Module):
  
  def __init__(self, input_size, hidden_size, lossfunc):
    super(MLP, self).__init__()

    self.linear_1 = nn.Linear(input_size, hidden_size)
    self.linear_2 = nn.Linear(hidden_size, hidden_size)
    self.linear_3 = nn.Linear(hidden_size, 1)
    self.lossfunc = lossfunc
    self.performance = []

  def set_optimizer(self):
    self.optimizer=torch.optim.SGD(self.parameters(), lr=1e-3)
    
  def forward(self, x):
    x1 = torch.sigmoid(self.linear_1(x))
    x2 = torch.sigmoid(self.linear_2(x1))
    return self.linear_3(x2)
    
  def get_performance(self, features, labels):
    
    features_torch = torch.from_numpy(features).float()
    labels_torch = torch.from_numpy(labels).float()
    pred = self.forward(features_torch)
    
    prediction = pred > 0.5
    prediction = torch.reshape(prediction, labels.shape)
    labels = labels_torch > 0.5

    TP = torch.sum(torch.logical_and(labels, prediction)).float()
    FP = torch.sum(torch.logical_and(torch.logical_not(labels), prediction)).float()
    TN = torch.sum(torch.logical_and(torch.logical_not(labels), torch.logical_not(prediction))).float()
    FN = torch.sum(torch.logical_and(labels, torch.logical_not(prediction))).float()
    print(TP, FP, TN, FN)
    self.performance.append({"Accuracy": (TP+TN)/(TP+TN+FP+FN),
                             "F1_score": (TP/(TP+FP)*TP/(TP+FN))})
  
  
  def train(self, features, labels, epoch):
    
    """
    Train a model for num_epoch epochs on the given data
    
    Inputs:
      self: an instance of nn.Module (or classes with similar signature)
      features: a numpy array
      labels: a numpy array
    """
    features_torch = torch.from_numpy(features).float()
    labels_torch = torch.from_numpy(labels).float()
    
    pred = self.forward(features_torch)
    loss = self.lossfunc(pred, labels_torch)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    
    prediction = pred > 0.5
    prediction = torch.reshape(prediction, labels.shape)
    labels = labels_torch > 0.5

    print ('Epoch [%d], Loss: %.4f' %(epoch, loss.item()))
        