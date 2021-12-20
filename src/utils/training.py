import pandas as pd
import numpy as np
from src.CONSTS import *
from src.utils.neural_network import *
from src.utils.make_batch import *
from src.utils.utils import shuffle_file_rows
from torch.utils.data import DataLoader

import time

def train():
  raw_data = load_data()
  input_data = transform_raw_data(raw_data)
  labels = get_labels(raw_data)
  model = create_model(NUM_OF_SEQUENCES)
  train_model(model,input_data, labels)
  
  return model


def load_data():
  data = pd.read_csv(train_test_sample_file_path,
                     names = ["name","start_seq", "end_seq", "labels"], sep=';')
  return data

def get_labels(df):
    """This function gets the labels defined in data[labels] and return as tensor"""
    labels = df["labels"].replace(0, -1)
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


def build_indices_batches(y, interval, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    k_fold = int(num_row / interval)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)
  

def create_model(input_size, hidden_size = 100):
  """This function creates a model""" 
  model = MLP(input_size = input_size, hidden_size = hidden_size, lossfunc=nn.HingeEmbeddingLoss())
  model.set_optimizer()
  return model


def train_model(model, X, labels, batch_size = 500, epoch = 10):
  """This funtion trains the model. First raw data is loaded,
  then for each batch this is translated. The model is trained 
  on these batches. This reapeted for n epochs"""
  
  start = time.time()
  
  for k in range(epoch):
    indices = build_indices_batches(labels, batch_size, seed = 2)
    
    # Train
    for i in range(len(indices)-1): # Last batch kept for performace evaluation
      x_batch = X[indices[i]]
      y_batch = labels[indices[i]]
      print("Training on batch ",i,"...")
      model.train(x_batch, y_batch, k)
    
    # Get performance
    x_batch = X[indices[-1]]
    y_batch = labels[indices[-1]]
    model.get_performance(x_batch, y_batch) 
    
    print("Epoch ",i," finished, total time taken:", time.time()-start)
    accuracy = model.performance[-1]["Accuracy"]
    F1score = model.performance[-1]["F1_score"]
    print("Accuracy is: %.4f and F1-score is: %.4f" %(accuracy, F1score))
     
    
    """If data is too big, replace this by:
    shuffle_file_rows(train_test_sample_file_path)
    raw_data = load_data()"""
    
    raw_data = raw_data.sample(frac=1).reset_index(drop=True)

    
