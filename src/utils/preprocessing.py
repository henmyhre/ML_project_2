from Bio import SeqIO
from src.CONSTS import * 
from src.utils.utils import *
import time
import numpy as np
import torch
 

def create_encoding_dictionaries(): 
  """"
  Creates protein encoding dictionaries for hot key encoding.
  The output is saved in the constants
  """
  for index, letter in enumerate(UNIQUE_CHARS):
    PROTEIN_ENCODING[letter] = [0 for _ in range(len(UNIQUE_CHARS))]
    PROTEIN_ENCODING[letter][index] = 1

    # Inverse dictionary
    BINARY_ENCODING[str(PROTEIN_ENCODING[letter])] = letter
  
  
def create_sparse_matrix_pytorch(device, df, cross_correlate = True):
    """
    This function creates a sparse matrix to use for training. These are very 
    sparse because the amino acid sequences are one-hot encoded. 
    The aminoacid sequences are first made into binary and added together. 
    Aftewards, nonzero entries indices and values are found.
    param: df: pd.DataFrame, containing the amino acid sequences in start_seq and end_seq
           cross_correlate: bool, if true, sequences are cross correlated
           
    retrun: output: sparse array.
    """
    
    print("Creating sparse matrix...")
    # TODO: add matrices while hot encoded or mutliply after dimension reduction?
    # Initialize list to save coordiantes and values of non-zero entries
    coo_matrix_rows = []
    coo_matrix_cols = []
    coo_matrix_data = []
    labels = []
    
    df["labels"] = df["labels"].replace(-1, 0)
    labels = []
    
    col_index = 0
    while not df.empty:   
        row = df.iloc[-1]   # Read last row
        labels.append(row["labels"])
        df = df.iloc[:-1]   # Delete last row to save memory
        labels.append(row["labels"].item())
        if cross_correlate:
            # one-hot encode amino acid sequences and add together
            one_hot = add_binary_sequences(seq_into_binary(row["start_seq"]),
                                        seq_into_binary(row["end_seq"]))
        else:
            # One hot encode amino acid sequences and put both ends into an array
            len_one_hot = len(row["end_seq"])*2*len(PROTEIN_ENCODING)
            one_hot = np.empty((1, len_one_hot))
            one_hot[0,:int(len_one_hot/2)] = seq_into_binary(row["start_seq"]).flatten()
            one_hot[0,int(len_one_hot/2):] = seq_into_binary(row["end_seq"]).flatten()
            one_hot = one_hot[0]
            
        # Find Nonzero indices
        non_zero = np.flatnonzero(one_hot)
        # Save row and col coordinates
        coo_matrix_cols.extend(non_zero.tolist())
        coo_matrix_rows.extend([col_index] * len(non_zero))
        # Save non zero values 
        coo_matrix_data.extend(one_hot[non_zero].tolist())
        # Increase col_index
        col_index +=1
        
        if col_index % 1000 == 0:
            print("At index", col_index)
            
    print("Putting into sparse...")
    # Create sparseamatrix
    factor_matrix = torch.sparse_coo_tensor([coo_matrix_rows, coo_matrix_cols], coo_matrix_data, device=device)
    return factor_matrix, torch.tensor(labels)

def seq_into_binary(sequence):
    """
    This function translates an aminoacid sequence into a one-hot encoded sequence
    param:  sequence: list
    return: encoded: ndarray
    """

    encoded = list()
    for letter in sequence:
        encoded.append(PROTEIN_ENCODING[letter])
    return np.array(encoded)


def add_binary_sequences(start, end):
    """This function adds hot encoded aminoacid positions together for each position in the sequence
    
    param:  start:  ndarray
            end:   ndarray
    return: output: ndarray
    """
    
    output = list()
    
    for i in range(len(start)):
        for j in range(len(end)):
            output.append(start[i] + end[j])
    
    return np.array(output).flatten()  
    

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
                    seq_false_file.write(name_and_seq + ";-1\n")

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
        split_raw_data_file()
        print(get_time_dif_str(start), "Done saving to files true false files. Shuffling...")
        
        shuffle_file_rows(seq_false_file_path)
        print(get_time_dif_str(start), "Done shuffling. Saving to preprocessed data...")
        
        create_train_test_data(false_per_true)
        print(get_time_dif_str(start), "Done saving finished preprocessed data. Shuffling...")
        
        shuffle_file_rows(train_test_sample_file_path)
        print(get_time_dif_str(start), "Done with shuffling. \n Done with preprocessing!:)")
        
        #df = save_encoded_data(train_test_sample_file_path, preprocessed_data_file_path)
        #print(get_time_dif_str(start), "Done with preprocessing!:)")
    else: print("Skipped saving to files.")

  

