import pandas as pd
import numpy as np
from src.CONSTS import *
from src.utils.neural_network import *
from src.utils.make_batch import *
import torch
import time

def train():
  raw_data = load_data()
  raw_data= raw_data.iloc[:500]
  # Create sparse matrix
  input_data = transform_raw_data(raw_data)
  labels = get_labels(raw_data)
  print(labels)
  # Create model, input size is size of feature lenght
  model = create_model(input_data.size()[1])
  # Train model
  train_model(model,input_data, labels)
  
  return model

def create_model(input_size, hidden_size = 100):
  """This function creates a model""" 
  model = MLP(input_size = input_size, hidden_size = hidden_size, lossfunc=nn.BCELoss())
  model.set_optimizer()
  return model


def load_data():
  data = pd.read_csv(train_test_sample_file_path,
                     names = ["name","start_seq", "end_seq", "labels"], sep=';')
  return data

def get_labels(df):
    """This function gets the labels defined in data[labels] and return as tensor"""
    labels = df["labels"].replace(-1, 0)
    print(labels.values)
    return torch.tensor(labels.values)
  

def transform_raw_data(data, reduce=False):
    """Takes raw data as input and outputs sparse matrix or reduced matrix.
    param: data: pd.DataFrame
           reduce: Bool
    return: output: ndarray or sparse array"""
  
    sparse = create_sparse_matrix_pytorch(data)
    if reduce:
        reduced = reduce_dimensionality(sparse)
        return reduced
    
    else:
        return sparse


def build_indices_batches(y, interval, seed=None):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    k_fold = int(num_row / interval)
    #np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return torch.tensor(k_indices).long()
  


def train_model(model, X, labels, batch_size = 100, epoch = 10):
  """This funtion trains the model. First raw data is loaded,
  then for each batch this is translated. The model is trained 
  on these batches. This reapeted for n epochs"""
  
  start = time.time()
    
  for k in tqdm(range(epoch)):
    # Different indices for test and training every round, "shuffles" the data
    indices = build_indices_batches(labels, batch_size, seed = 2)
    # Train
    for i in range(indices.size()[0] - 1): # Last batch kept for performace evaluation
      x_batch = X.index_select(0, indices[i,:]).to_dense()  # Get dense representation
      y_batch = labels.index_select(0, indices[i,:])
      print("Training on batch ",i,"...")
      model.train(x_batch, y_batch, k)
    
    
    # Get performance
    x_batch = X.index_select(0, indices[-1,:]).to_dense()
    y_batch = labels.index_select(0, indices[-1,:])
    model.get_performance(x_batch, y_batch) 
    
    print("Epoch ",i," finished, total time taken:", time.time()-start)
    accuracy = model.performance[-1]["Accuracy"]
    F1score = model.performance[-1]["F1_score"]
    print("Accuracy is: %.4f and F1-score is: %.4f" %(accuracy, F1score))
     
    
    """If data is too big, replace this by:
    shuffle_file_rows(train_test_sample_file_path)
    raw_data = load_data()"""
    

    
