import numpy as np
from src.CONSTS import *
from src.utils.classifier import *
from src.utils.preprocessing import *
from src.utils.model_utils import *
import time

import torch
import matplotlib.pyplot as plt


def create_model(input_size, hidden_size = 100):
    """This function creates a model""" 
    model = BinaryClassfier(input_size = input_size, hidden_size = hidden_size)
    return model


def build_indices_batches(y, interval, seed=None):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    k_fold = int(num_row / interval)
    #np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return torch.tensor(k_indices).long()


def train(model, X, labels, batch_size = 500, epoch = 100, lr=1e-2, lossfunc=nn.BCEWithLogitsLoss()):
    """This funtion trains the model. First raw data is loaded,
    then for each batch this is translated. The model is trained 
    on these batches. This reapeted for n epochs"""
    
    # initialize optimzer and lossfunc
    loss_fn = lossfunc
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Set to training mode
    model.train()
    losses = list()
    
    start = time()
    
    for k in range(epoch):
        # Different indices for test and training every round, "shuffles" the data
        indices = build_indices_batches(labels, batch_size)
        
        # Train
        for i in range(indices.size()[0]): # Last batch kept for performace evaluation
            x_batch = X[indices[i,:]].float()
            y_batch = labels[indices[i,:]].float()
            #x_batch = X.index_select(0, indices[i,:]).to_dense().float()  # Get dense representation
            #y_batch = labels.index_select(0, indices[i,:]).float()
            
            # set optimizer to zero grad
            optimizer.zero_grad()   
            # forward pass
            y_pred = model.forward(x_batch)
            y_pred = y_pred.reshape(y_batch.size())
            # evaluate
            loss = loss_fn(y_pred, y_batch)
            losses.append(loss.item())
            # backward pass
            loss.backward()
            optimizer.step()
            
        print("Epoch ",k," finished, total time taken:", time.time()-start)
        
    plt.plot(np.array(losses))
    
    
            

        

        
