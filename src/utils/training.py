import pandas as pd
import numpy as np
from src.CONSTS import *
from src.utils.classifier import *
from src.utils.preprocessing import *
import torch
import time
from sklearn.metrics import accuracy_score, f1_score


def train(gpu = False):
    
    device = torch.device('cpu')
    if gpu:
        device = torch.device('cuda')
        
    raw_data = load_data()
    # Create sparse matrix
    input_data, labels = create_sparse_matrix_pytorch(device, raw_data)
    labels = labels.to(device=device)
    #labels = get_labels(raw_data).to(device=device)
    # Create model, input size is size of feature lenght
    model = create_model(device, input_data.size()[1])
    # Train model
    train_model(device, model, input_data, labels)  
    return model


def create_model(device, input_size, hidden_size = 100):
    """This function creates a model""" 
    model = BinaryClassfier(input_size = input_size, hidden_size = hidden_size).to(device=device)
    return model


def load_data():
    data = pd.read_csv(train_test_sample_file_path,
                        names = ["name","start_seq", "end_seq", "labels"], sep=';')
    return data


def get_labels(df):
    """This function gets the labels defined in data[labels] and return as tensor"""
    labels = df["labels"].replace(-1, 0)
    return torch.tensor(labels.values)


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
    y_pred = np.round(y_pred.cpu().detach().numpy())
    
    accuracy = accuracy_score(y_true, y_pred)
    F_score = f1_score(y_true, y_pred)
    return accuracy, F_score


def train_model(device, model, X, labels, batch_size = 100, epoch = 10, lr=1e-6, lossfunc=nn.BCELoss()):
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
        indices = build_indices_batches(labels, batch_size).to(device=device)
        # Train
        for i in range(indices.size()[0] - 1): # Last batch kept for performace evaluation
            print("Training on batch ",i,"...")
            x_batch = X.index_select(0, indices[i,:]).to_dense().float().to(device=device)  # Get dense representation
            y_batch = labels.index_select(0, indices[i,:]).float().to(device=device)
            
            # set optimizer to zero grad
            optimizer.zero_grad()   
            # forward pass
            y_pred = model.forward(x_batch).to(device=device)
            y_pred = y_pred.reshape(y_batch.size()).to(device=device)
            # evaluate
            loss = loss_fn(y_pred, y_batch)
            losses.append(loss.item())
            print(loss.item())
            # backward pass
            loss.backward()
            optimizer.step()

        # Get performance after epoch
        x_batch = X.index_select(0, indices[-1,:]).to_dense().float().to(device=device)  # Get dense representation
        y_batch = labels.index_select(0, indices[-1,:]).float().to(device=device)
        # get pred
        y_pred = model.forward(x_batch).to(device=device)
        y_pred = y_pred.reshape(y_batch.size()).to(device=device)
        # Get metrics
        accuracy, F_score = get_performance(y_batch, y_pred)
        
        
        print("Epoch ",i," finished, total time taken:", time.time()-start)
        print("Accuracy is: %.4f and F1-score is: %.4f" %(accuracy, F_score))
            

        

        
