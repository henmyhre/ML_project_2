from Bio import SeqIO
import csv
import itertools
import pandas as pd
import numpy as np
from consts import * 
 

def create_encoding_dictionaries():
  
  unique_chars = ['V', 'W', 'H', 'G', 'T', 'Q', 'A', 'C', 'S', 'K', 'Y', 'L', 'R', 'D', 'M', 'E', 'I', 'F', 'N', 'P', '-', 'X']
  
  for index, letter in enumerate(unique_chars):
    PROTEIN_ENCODING[letter] = [0 for _ in range(len(unique_chars))]
    PROTEIN_ENCODING[letter][index] = 1

    # Inverse dictionary
    BINARY_ENCODING[str(PROTEIN_ENCODING[letter])] = letter


def split_sequence(sequence):
  seq_len = len(sequence)
  start, end = sequence[:int(seq_len/2)], sequence[int(seq_len/2):]
  return start, end


def split_file(input_file, output_file):
  
  fasta_sequences = SeqIO.parse(open(input_file),'fasta')
  
  seq_start_list = []
  seq_end_list = []
  
 
  for fasta in fasta_sequences:
    name, sequence = fasta.id, str(fasta.seq)
    sequence_start, sequence_end = split_sequence(sequence)
    seq_start_list.append([name, sequence_start])
    seq_end_list.append([name, sequence_end])
    
  count_true = 0
  count_false = 0
  
  for index, start in enumerate(seq_start_list):
    file = output_file.split('.')[0] + str(index)+'.csv'
    
    with open(file, "w") as seq_file:    
      for index, end in enumerate(seq_end_list):
        label = 0
        if start[0] == end[0]: 
          label = 1
          count_true += 1
        else:
          count_false += 1
        seq_file.write(start[0] + "<>" + end[0] + ";" + start[1] + ";" + end[1] + ";" + str(label) + "\n")
        
    seq_file.close()
        
  return count_true, count_false


def get_csv_line(path, line_number):
  with open(path) as f:
    return next(itertools.islice(csv.reader(f), line_number, None))


def load_batch(path, file_indices, row_indices, batch_size):
    
  batch = pd.DataFrame(columns = ["name", "first half", "second half", "label"])
  for i in range(batch_size):
      
    file = path.split('.')[0] + str(file_indices[i])+'.csv'
    line = get_csv_line(file, row_indices[i])
    
    name, start, end, label = line[0].split(';')
    start_encoded = into_binary(start)
    end_encoded = into_binary(end)
    
    batch = batch.append({"name" : name, "first half":start_encoded,
                          "second half":end_encoded, "label":label}, ignore_index=True)   
  return batch

    
def into_binary(sequence):
  encoded = list()
  for letter in sequence:
    encoded.append(PROTEIN_ENCODING[letter])
  
  return np.array(encoded)


def compare_sequences(start, end):
    
  output = list()
  
  for i in range(len(start)):
    for j in range(len(end)):
      output.append(start[i] + end[j])
  
  return np.array(output).flatten()


def create_batch_input(batch):
  batch_input = list()
  
  for index in range(len(batch)):
    batch_input.append(compare_sequences(batch["first half"][index], batch["second half"][index]))
  
  return np.array(batch_input) 


def build_k_indices(num_row, k_fold, shape, seed):
  """build k indices for k-fold."""
  interval = int(num_row / k_fold)
  np.random.seed(seed)
  indices = np.random.permutation(num_row)
  
  k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
  np.array(k_indices)
  
  return np.unravel_index(k_indices, shape)
  