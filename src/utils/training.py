import numpy as np
from src.CONSTS import *
from src.utils.classifier import *
from src.utils.model_utils import *
import time
from src.utils.test import test
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


def train(model, input_data, labels, batch_size = 500, epoch = 20, lr=1e-2, lossfunc=nn.BCEWithLogitsLoss()):
    """This funtion trains the model. First raw data is loaded,
    then for each batch this is translated. The model is trained 
    on these batches. This reapeted for n epochs"""
    # split data
    split_size = int(input_data.size()[0]*0.8)
    train_input_data, test_input_data = torch.split(input_data, split_size)
    train_labels, test_labels = torch.split(labels, split_size)
    
    # initialize optimzer and lossfunc
    loss_fn = lossfunc
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Set to training mode
    model.train()
    
    # save performance
    losses = list()
    accuracies = list()
    f_scores = list()
    
    # Track speed
    start = time.time()
    
    for k in range(epoch):
        # Different indices for test and training every epoch, "shuffles" the data
        indices = build_indices_batches(train_labels, batch_size)
        
        # Train
        for i in range(indices.size()[0]): # Last batch kept for performace evaluation
            x_batch = train_input_data[indices[i,:]].float()
            y_batch = train_labels[indices[i,:]].float()
            
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
        
        accuracy, F_score = test(model, test_input_data, test_labels)
        accuracies.append(accuracy)
        f_scores.append(F_score)
        print("Epoch ",k," finished, total time taken:", time.time()-start)
      
    plt.plot(np.array(losses))
    plt.show()
    plt.plot(np.array(accuracies))
    plt.show()
    plt.plot(np.array(f_scores))
    plt.show()
    
    print("Final accuracy is: %.4f and final F-score is %.4f" %(accuracy, F_score))
    
    
            

        

        
