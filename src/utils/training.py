import numpy as np
import torch
import torch.nn as nn
import time
from src.CONSTS import *
from src.utils.classifier import *
from src.utils.model_utils import *


def create_model(input_size, model_type = NEURAL_NET_1, hidden_size_1 = 100, hidden_size_2 = 20):
    """
    This function creates a learning model, a one layered neural network, atwo layered neural network
    or a logistic regressor.
    
    param: input_size: int, size of the input vector
           model_type: str, "One layer neural net", "Two layer neural net" or "Logistic regression"
           hidden_size_1: int, size of first hidden layer, if model_type is a neural net
           hidden_size_2: int, size of second hidden layer, if mode_type is a two layered neural net
           
    return: model: Module
    """ 
    
    if model_type == NEURAL_NET_2:
        model = BinaryClassfier_two_layer(input_size, hidden_size_1, hidden_size_2)
        
    elif model_type == NEURAL_NET_1:
        model = BinaryClassfier_one_layer(input_size, hidden_size_1)
        
    else:
        model = LogisticRegression(input_size)
        
    return model


def build_indices_batches(y, interval):
    """
    Build indices for the batches. Creates random groups of indices with groupsize of interval and maximum index
    equal to the length of y.
    param: y: torch.tensor
           interval: int
    return: torch.tensor.long
    """
    
    num_row = y.shape[0]
    k_fold = int(num_row / interval)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return torch.tensor(k_indices).long()


def create_weights(y_pred, true_weight):
    w = y_pred.clone()
    w[w==0] = 1
    w[w==1] = true_weight
    return w


def train(model, input_data, labels, model_name = None, batch_size = 1000, epoch = 100, lr=1e-2):
    """
    This funtion trains the model. First raw data is loaded,
    then for each batch this is translated. The model is trained 
    on these batches. This reapeted for n epochs.
    
    param: model: Module
           input_data: torch.tensor
           labels: torch.tensor
           model_name: str, save performance measures under this name
           batch_size: int
           epoch: int
           lr: float, learning rate used
    return :float, final F1-score
           :float, final accuracy
    """
    
    # split data
    split_size = int(input_data.size()[0]*0.8)
    train_input_data, test_input_data = torch.split(input_data, split_size)
    train_labels, test_labels = torch.split(labels, split_size)
    
    # initialize optimzer and lossfunc
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
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
            y_batch = y_batch.reshape(y_pred.size())
                    
            # evaluate
            loss = loss_fn(y_pred, y_batch)
            
            # Safe loss
            losses.append(loss.item())
            
            # backward pass
            loss.backward()
            optimizer.step()
        
        accuracy, F_score = test(model, test_input_data, test_labels)
        accuracies.append(accuracy)
        f_scores.append(F_score)
        #print("Epoch ",k," finished, total time taken:", time.time()-start)
    
    # Show performance
    plot_result(losses, 'training step', 'Loss', model_name)
    plot_result(accuracies, 'Epoch', 'Accuracy', model_name)
    plot_result(f_scores, 'Epoch', 'F1-score', model_name)
 
    print("Final accuracy is: %.4f and final F-score is %.4f" %(accuracy, F_score))
    print("Total time was %.4f" %(time.time()-start))
    print(model_name)
    
    return f_scores[-1], accuracies[-1]
    
    
            

        

        
