import pandas as pd
import numpy as np
from src.CONSTS import *
from src.utils.neural_network import *
from src.utils.make_batch import create_batch
from src.utils.utils import shuffle_file_rows

import time

def train():
  raw_data = load_data()
  model = create_model(5353)
  train_model(model,raw_data)
  
  return model



def load_data():
  
  data = pd.read_csv(train_test_sample_file_path,
                     names = ["name","start_seq", "end_seq", "labels"], sep=';')
  return data


def create_model(input_size, hidden_size = 100):
  """This function creates a model"""
  
  model = MLP(input_size = input_size, hidden_size = hidden_size, lossfunc=nn.HingeEmbeddingLoss())
  model.set_optimizer()
  return model


def train_model(model, raw_data, batch_size = 500, epoch = 10):
  """This funtion trains the model. First raw data is loaded,
  then for each batch this is translated. The model is trained 
  on these batches. This reapeted for n epochs"""
  
  start = time.time()
  
  for k in range(epoch):
    
    steps = np.linspace(0, int(len(raw_data)), int(len(raw_data)/batch_size), dtype = int)
    
    # Train
    for i in range(len(steps)-2): # Last batch kept for performace evaluation
      print("Make batch ",i,"...")
      x_batch, y_batch = create_batch(raw_data, steps[i], steps[i+1])
      print("Training on batch ",i,"...")
      model.train(x_batch, y_batch, k)
    
    # Get performance
    x_batch, y_batch = create_batch(raw_data, steps[i], steps[i+1])
    model.get_performance(x_batch, y_batch) 
    
    print("Epoch ",i," finished, total time taken:", time.time()-start)
    accuracy = model.performance[-1]["Accuracy"]
    F1score = model.performance[-1]["F1_score"]
    print("Accuracy is: %.4f and F1-score is: %.4f" %(accuracy, F1score))
     
    
    """If data is too big, replace this by:
    shuffle_file_rows(train_test_sample_file_path)
    raw_data = load_data()"""
    
    raw_data = raw_data.sample(frac=1).reset_index(drop=True)

    
