import pandas as pd
import numpy as np
from src.CONSTS import *
import csv
from src.utils.neural_network import *
import time

def load_batch(path, row_index):
    
  batch = pd.DataFrame(columns = ["name", "first half", "second half", "label"])
  
  reader = pd.read_csv(preprocessed_data_file_path, sep=';', chunksize=10, iterator=True)

  for row in reader:
    name, start, end, label = row[0].split(';')
    start_encoded = seq_into_binary(start)
    end_encoded = seq_into_binary(end)

    batch = batch.append({"name" : name, "first half":start_encoded,
                          "second half":end_encoded, "label":label}, ignore_index=True)   
  return batch


def seq_into_binary(sequence):
  encoded = list()
  for letter in sequence:
    encoded.append(PROTEIN_ENCODING[letter])
  
  return np.array(encoded)


def add_binary_sequences(start, end):
    
  output = list()
  
  for i in range(len(start)):
    for j in range(len(end)):
      output.append(start[i] + end[j])
  
  return np.array(output).flatten()


def create_batch_input(batch):
  batch_input = list()
  
  for index in range(len(batch)):
    batch_input.append(add_binary_sequences(batch["first half"][index], batch["second half"][index]))
  
  return np.array(batch_input) 


def create_model():
  model = MLP(hidden_size=2, lossfunc=nn.MSELoss())
  model.set_optimizer()
  return model


def train_model(model):
  batch_size = 500
  start = time.time()
  
  #for batch in pd.read_csv(preprocessed_data_file_path, chunksize = batch_size):
    
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
