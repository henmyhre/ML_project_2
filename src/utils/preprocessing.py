from Bio import SeqIO
import csv
import itertools
import pandas as pd
import numpy as np
from src.CONSTS import * 
from csv import reader
import random
import shutil
from utils.utils import *
import os
 

def create_encoding_dictionaries(): 
  for index, letter in enumerate(UNIQUE_CHARS):
    PROTEIN_ENCODING[letter] = [0 for _ in range(len(UNIQUE_CHARS))]
    PROTEIN_ENCODING[letter][index] = 1

    # Inverse dictionary
    BINARY_ENCODING[str(PROTEIN_ENCODING[letter])] = letter


def split_sequence(sequence):
  seq_len = len(sequence)
  start, end = sequence[:int(seq_len/2)], sequence[int(seq_len/2):]
  return start, end
    

def split_file():
  
  fasta_sequences = SeqIO.parse(open(raw_data_file_path),'fasta')
  
  seq_start_list = []
  seq_end_list = []
  
  for fasta in fasta_sequences:
    name, sequence = fasta.id, str(fasta.seq)
    sequence_start, sequence_end = split_sequence(sequence)
    seq_start_list.append([name, sequence_start])
    seq_end_list.append([name, sequence_end])
  
  with open(seq_true_file_path, "w") as seq_true_file, open(seq_false_file_path, "w") as seq_false_file: 
    for start in seq_start_list:   
      for end in seq_end_list:
        
        name_and_seq = start[0] + "<>" + end[0] + ";" + start[1] + ";" + end[1]
        if start[0] == end[0]: 
          seq_true_file.write(name_and_seq + ";1\n")
        else:
          seq_true_file.write(name_and_seq + ";0\n")

      import random
    
    seq_true_file.close()
    seq_false_file.close()


def shuffle_file_rows(file):
  with open(file, "w") as f:
    lines = f.readlines()
    random.shuffle(lines)
    f.writelines(lines)
    file.close()
  
    
def create_train_test_data(false_per_true):
  
  false_amount = num_of_sequences*false_per_true
  
  clear_file(preprocessed_data_file_path)
  
  shutil.copyfile(seq_true_file_path, preprocessed_data_file_path)

  row_size = 1000
  for i in range(1, false_amount, row_size):
    df = pd.read_csv(seq_false_file_path,
          header=None,
          nrows = row_size,
          skiprows = i)
    df.to_csv(preprocessed_data_file_path,
          index=False,
          header=False,
          mode='a',
          chunksize=row_size)
    
  
def preprocessing(force_save_seq = False, false_per_true = 10):
  if (len(BINARY_ENCODING) == 0) or (len(PROTEIN_ENCODING) == 0):  
    # Create dictionaries for encoding proteins
    create_encoding_dictionaries()
    
  # Create CSV files and get size of true and false data
  if (os.stat(preprocessed_data_file_path).st_size == 0) or force_save_seq:
    print("Starting saving to files")
    split_file()
    print("Done saving to files. Shuffling...")
    
    shuffle_file_rows(seq_true_file_path)
    shuffle_file_rows(seq_false_file_path)
    print("Done shuffling.")
    
    create_train_test_data(false_per_true)
    
  else: print("Skipped saving to files.")
