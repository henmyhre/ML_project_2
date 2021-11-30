# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:38:30 2021

@author: jurri
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden_size):
        super(MLP, self).__init__()
        # TODO: Define parameters / layers of a multi-layered perceptron with one hidden layer
        self.linear_1 = nn.Linear(1, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # TODO: implement forward computation
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
    # TODO: Step 1 - create torch variables corresponding to features and labels
    features_torch = torch.from_numpy(features)
    labels_torch = torch.from_numpy(labels)
    dataset = torch.utils.data.TensorDataset(features_torch, labels_torch)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    for epoch in range(num_epoch):
        
        for id_batch, (x_batch, y_batch) in enumerate(dataloader):
            
            # TODO: Step 2 - compute model predictions and loss
            pred = model.forward(x_batch)
            loss = lossfunc(pred, y_batch)

            # TODO: Step 3 - do a backward pass and a gradient update step
            # Hint: Do not forget to first clear the gradients from the previous rounds
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        
        if epoch % 10 == 0:
            print ('Epoch [%d/%d], Loss: %.4f' %(epoch+1, num_epoch, loss.item()))
        