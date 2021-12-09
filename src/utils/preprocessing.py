from Bio import SeqIO
from src.CONSTS import * 
from src.utils.utils import *
import time
import pandas as pd
import tqdm
import numpy as np
 

def create_encoding_dictionaries(): 
  """"
  Creates protein encoding dictionaries fro hot key encoding.
  """
  for index, letter in enumerate(UNIQUE_CHARS):
    PROTEIN_ENCODING[letter] = [0 for _ in range(len(UNIQUE_CHARS))]
    PROTEIN_ENCODING[letter][index] = 1

    # Inverse dictionary
    BINARY_ENCODING[str(PROTEIN_ENCODING[letter])] = letter
    

def split_raw_data_file():
  """
  Splits the raw data file and saves true and false combinations of sequences in separate files.
  """
  
  fasta_sequences = SeqIO.parse(open(raw_data_file_path),'fasta')
  
  seq_start_list = []
  seq_end_list = []
  
  for fasta in fasta_sequences:
    name, sequence = fasta.id, str(fasta.seq)
    sequence_start, sequence_end = split_sequence(sequence)
    seq_start_list.append([name, sequence_start])
    seq_end_list.append([name, sequence_end])
  
  
  with open(seq_true_file_path, "w+") as seq_true_file, open(seq_false_file_path, "w+") as seq_false_file: 
    for start in seq_start_list:   
      for end in seq_end_list:
        
        name_and_seq = start[0] + "<>" + end[0] + ";" + start[1] + ";" + end[1]
        if start[0] == end[0]: 
          seq_true_file.write(name_and_seq + ";1\n")
        else:
          seq_false_file.write(name_and_seq + ";0\n")

    seq_true_file.close()
    seq_false_file.close()
  

def create_train_test_data(false_per_true):
  """
  Saves all true data and a random sample of the false data to a file for training and testing.
  """
  
  true_amount = NUM_OF_SEQUENCES
  false_amount = NUM_OF_SEQUENCES*false_per_true
  
  clear_file(train_test_sample_file_path)
  
  append_csv_rows_to_new_csv(seq_true_file_path, train_test_sample_file_path, true_amount)
  append_csv_rows_to_new_csv(seq_false_file_path, train_test_sample_file_path, false_amount)


def preprocessing(force_save_seq = False, false_per_true = 1):
  """
  From raw data file to file with random train/test data in random order.
  """
  if (len(BINARY_ENCODING) == 0) or (len(PROTEIN_ENCODING) == 0):  
    # Create dictionaries for encoding proteins
    create_encoding_dictionaries()
    
  if force_save_seq:
    start = time.time()
    
    print(get_time_dif_str(start), "Starting saving to files")
    #split_raw_data_file()
    print(get_time_dif_str(start), "Done saving to files true false files. Shuffling...")
    
    #shuffle_file_rows(seq_false_file_path)
    print(get_time_dif_str(start), "Done shuffling. Saving to preprocessed data...")
    
    #create_train_test_data(false_per_true)
    print(get_time_dif_str(start), "Done saving finished preprocessed data. Shuffling...")
    
    #shuffle_file_rows(train_test_sample_file_path)
    print(get_time_dif_str(start), "Done with shuffling. \n Done with preprocessing!:)")
    
    #df = save_encoded_data(train_test_sample_file_path, preprocessed_data_file_path)
    #print(get_time_dif_str(start), "Done with preprocessing!:)")
  else: print("Skipped saving to files.")

  
  
"""Probably wont use this""" 
  

# =============================================================================
# def seq_into_binary(sequence):
#   encoded = list()
#   for letter in sequence:
#     encoded.append(PROTEIN_ENCODING[letter])
#   
#   return np.array(encoded)
# 
# 
# def add_binary_sequences(start, end):
#     
#   output = list()
#   
#   for i in range(len(start)):
#     for j in range(len(end)):
#       output.append(start[i] + end[j])
#   
#   return np.array(output).flatten()
# 
# 
# def save_encoded_data(input_file, output_file):
#   df = pd.read_csv(input_file, sep=";", names = ["names", "start_seq", "end_seq", "label"])
#   list_to_save = {"names": df["names"].to_list(), "seq_encoded": list(), "label": df["label"].to_list()}
#   df.drop("names", axis = 1)
#   df.drop("label", axis = 1)
#   start_seq = df["start_seq"].to_list()
#   end_seq = df["start_seq"].to_list()
#   del df
#   i = 0
#   while len(start_seq):
#     i += 1
#     start_encoded = seq_into_binary(start_seq[0])
#     del start_seq[0]
#     end_encoded = seq_into_binary(end_seq[0])
#     del end_seq[0]
#     list_to_save["seq_encoded"].append(add_binary_sequences(start_encoded, end_encoded).tolist())
#     
#     if i and i % 1000 == 0:
#         print("At row", i)
#         
# 
#         
# import json
# with open('data/encoded.json', 'w') as fp:
#     json.dump(list_to_save, fp)
# 
# 
# =============================================================================

