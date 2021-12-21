import pandas as pd
import numpy as np
from src.CONSTS import *
from src.utils.classifier import *
from src.utils.preprocessing import *
from src.utils.preprocess_pca import transform_data

import torch
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

def train():
        
    raw_data = load_data()
    # Create sparse matrix
    input_data, labels = transform_data(raw_data)
    # Create model, input size is size of feature lenght
    model = create_model(input_data.size()[1])
    # Train model
    train_model(model, input_data, labels)  
    return model


def create_model(input_size, hidden_size = 100):
    """This function creates a model""" 
    model = BinaryClassfier(input_size = input_size, hidden_size = hidden_size)
    return model


def load_data():
    data = pd.read_csv(train_test_sample_file_path,
                        names = ["name","start_seq", "end_seq", "labels"], sep=';')
    return data



def build_indices_batches(y, interval, seed=None):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    k_fold = int(num_row / interval)
    #np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return torch.tensor(k_indices).long()


def get_performance(y_true, y_pred):
        
    y_true = y_true.cpu().numpy()
    
    y_pred = y_pred.cpu().detach()
    sig = torch.nn.Sigmoid()
    y_pred = sig(y_pred)
    y_pred = np.round(y_pred.numpy())
    accuracy = accuracy_score(y_true, y_pred)
    F_score = f1_score(y_true, y_pred)
    return accuracy, F_score


def train_model(model, X, labels, batch_size = 500, epoch = 100, lr=1e-2, lossfunc=nn.BCEWithLogitsLoss()):
    """This funtion trains the model. First raw data is loaded,
    then for each batch this is translated. The model is trained 
    on these batches. This reapeted for n epochs"""
    
    # initialize optimzer and lossfunc
    loss_fn = lossfunc
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Set to training mode
    model.train()
    
    start = time.time()
    losses =list()
    
    for k in range(epoch):
        # Different indices for test and training every round, "shuffles" the data
        indices = build_indices_batches(labels, batch_size)
        
        # Train
        for i in range(indices.size()[0] - 1): # Last batch kept for performace evaluation
            print("Training on batch ",i,"...")
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
            print(loss.item())
            # backward pass
            loss.backward()
            optimizer.step()

        # Get performance after epoch
        x_batch = X[indices[-1,:]].float()
        y_batch = labels[indices[-1,:]].float()
       # x_batch = X.index_select(0, indices[-1,:]).to_dense().float()  # Get dense representation
       # y_batch = labels.index_select(0, indices[-1,:]).float()
        # get pred
        y_pred = model.forward(x_batch)
        y_pred = y_pred.reshape(y_batch.size())
        # Get metrics
        accuracy, F_score = get_performance(y_batch, y_pred)
        
        
        print("Epoch ",k," finished, total time taken:", time.time()-start)
        print("Accuracy is: %.4f and F1-score is: %.4f" %(accuracy, F_score))
        
    plt.plot(np.array(losses))
    
    
            

        

        
