# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 18:45:27 2021

@author: jurri
"""


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
    return factor_matrix.to(device=device), torch.tensor(labels).to(device=device)

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