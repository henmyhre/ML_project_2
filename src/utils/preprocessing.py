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
    
def create_sparse_matrix_pytorch(df, cross_correlate = True):
    """This function creates a sparse matrix for all the raw input vectors. These are very 
    sparse because they are one-hot encoded.
    param: df: pd.DataFrame, containing the amino acid sequences in start_seq and end_seq
    retrun: output: sparse array.
    """
    print("Creating sparse matrix...")
    # TODO: add matrices while hot encoded or mutliply after dimension reduction?
    coo_matrix_rows = []
    coo_matrix_cols = []
    coo_matrix_data = []
    index = 0
    while not df.empty:   
        row = df.iloc[-1]     #Delete last row when it gets read out
        df = df.iloc[:-1]
    # for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if cross_correlate:
            one_hot = add_binary_sequences(seq_into_binary(row["start_seq"]),
                                        seq_into_binary(row["end_seq"]))
        else:
            one_hot = np.empty((1, row["end_seq"]*2))
            one_hot[0,:len(row["start_seq"])] = row["start_seq"]
            one_hot[0,len(row["end_seq"]):] = row["end_seq"]
        
        # Find Nonzero indices
        non_zero= np.flatnonzero(one_hot)
        # save row and col coordinates
        coo_matrix_cols.extend(non_zero.tolist())
        coo_matrix_rows.extend([index] * len(non_zero))
        # Non zero values 
        coo_matrix_data.extend(one_hot[non_zero].tolist())
        # Increase index
        index +=1
        if index % 1000 == 0:
            print("At index", index)
    print("Putting into sparse...")
    # Create sparseamatrix
    factor_matrix = torch.sparse_coo_tensor([coo_matrix_rows, coo_matrix_cols], coo_matrix_data)
    return factor_matrix
  
    

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

  

