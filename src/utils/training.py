import pandas as pd
import numpy as np
from src.CONSTS import *
from src.utils.neural_network import *
from src.utils.make_batch import create_batch
from src.utils.utils import shuffle_file_rows

import time

def train():
  raw_data = load_data()
  model = create_model(len(PROTEIN_ENCODING) * len(raw_data["start_seq"][0])**2)
  train_model(model,raw_data)
  
  return model



def load_data():
  
  data = pd.read_csv(train_test_sample_file_path,
                     names = ["name","start_seq", "end_seq", "labels"], sep=';')
  return data


def create_model(input_size, hidden_size = 100):
  """This function creates a model"""
  
  model = MLP(input_size = input_size, hidden_size = hidden_size, lossfunc=nn.MSELoss())
  model.set_optimizer()
  return model


def train_model(model, raw_data, batch_size = 500, epoch = 5):
  """This funtion trains the model. First raw data is loaded,
  then for each batch this is translated. The model is trained 
  on these batches. This reapeted for n epochs"""
  
  start = time.time()
  
  for k in range(epoch):
    
    steps = np.linspace(0, int(len(raw_data)), int(len(raw_data)/batch_size), dtype = int)
    
    # Train
    for i in range(len(steps)-2): # Last batch kept for performace evaluation
      print("Training on batch",i," ...")
      x_batch, y_batch = create_batch(raw_data, steps[i], steps[i+1])
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

    
  
  
  
  
# =============================================================================
#   batch = pd.DataFrame(columns = ["name", "first half", "second half", "label"])
#   
#   reader = pd.read_csv(preprocessed_data_file_path, sep=';', chunksize=1000, iterator=True)
# 
#   for row in reader:
#     name, start, end, label = row[0].split(';')
#     start_encoded = seq_into_binary(start)
#     end_encoded = seq_into_binary(end)
# 
#     batch = batch.append({"name" : name, "first half":start_encoded,
#                           "second half":end_encoded, "label":label}, ignore_index=True)   
# =============================================================================  
  
  
  
  
# =============================================================================
    # df = pd.read_csv(preprocessed_data_file_path, sep=';') #TODO: maybe add iterator
#   batches = []
#   batch = []
#   
#   for row in df:
#     if (row.index // 1000 == 0): 
#       batches.append(batch)
#       batch = []
#       
#     names, start, end, label = row[0].split(';')
#     start_encoded = seq_into_binary(start)
#     end_encoded = seq_into_binary(end)
# 
#     batch.append({"names" : names, "first half":start_encoded,
#                           "second half":end_encoded, "label":label}, ignore_index=True)   
#   
#   del df
#   
# =============================================================================
  
  
  

  """
  for i, file_index in enumerate(file_indices):
    print("Index", i, "\nFile", file_index, "\nTime taken:", int(time.time() - start), "seconds")
    batch = load_batch(seq_file_path, file_index)
    batch_input = create_batch_input(batch)
    labels = batch["label"].astype(int).to_numpy()
  
  end = time.time()
  print("Training time: " + str(end - start))

  model.train(batch_input, labels, 1, batch_size)
"""
