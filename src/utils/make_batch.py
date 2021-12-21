from src.CONSTS import * 
import pandas as pd
import numpy as np
from src.CONSTS import *
import umap
import time
import torch
import scipy.sparse
from tqdm import tqdm
import sympy
import sklearn.datasets
import sklearn.feature_extraction.text
import umap
import umap.plot
import matplotlib.pyplot as plt


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

  
def create_sparse_matrix_scipy(df, cross_correlate = True):
    """This function creates a sparse matrix for all the raw input vectors. These are very 
    sparse because they are one-hot encoded.
    param: df: pd.DataFrame, containing the amino acid sequences in start_seq and end_seq
    retrun: output: sparse array.
    """
    print("Creating sparse matrix...")
    # TODO: add matrices while hot encoded or mutliply after dimension reduction?
    lil_matrix_rows = []
    lil_matrix_data = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        if cross_correlate:
            one_hot = add_binary_sequences(seq_into_binary(row["start_seq"]),
                                            seq_into_binary(row["end_seq"]))
        else:
            one_hot = np.empty((1, row["end_seq"]*2))
            one_hot[0,:len(row["start_seq"])] = row["start_seq"]
            one_hot[0,len(row["end_seq"]):] = row["end_seq"]
        
        # Find Nonzero indices
        non_zero= np.flatnonzero(one_hot)
        lil_matrix_rows.append(non_zero.tolist())
        # Non zero values 
        lil_matrix_data.append(one_hot[non_zero].tolist())
    
    # Create sparseamatrix
    factor_matrix = scipy.sparse.lil_matrix((len(lil_matrix_rows), len(one_hot)), dtype=np.float32)
    factor_matrix.rows = np.array(lil_matrix_rows)
    factor_matrix.data = np.array(lil_matrix_data)
    
    return factor_matrix
  
  
def seq_into_binary(sequence):
    """This function translates an aminoacid sequence into a one-hot encoded sequence
    param:  sequence: list
    return: encoded: ndarray"""

    encoded = list()
    for letter in sequence:
        encoded.append(PROTEIN_ENCODING[letter])
    return np.array(encoded)


def add_binary_sequences(start, end):
    """This function adds hot encoded aminoacid positions together for each position in the sequence
    
    param:  start:  ndarray
            end:   ndarray
    return: output: ndarray"""
    
    output = list()
    
    for i in range(len(start)):
        for j in range(len(end)):
            output.append(start[i] + end[j])
    
    return np.array(output).flatten()


def reduce_dimensionality(sparsematrix):
    """This function reduces the dimension of the sparsematrix"""
    # Fit reduce model on 10% of data
    size = int(list(sparsematrix.shape)[0]*0.1)
    fit_data = sparsematrix[:size, :]
    mapper = umap.UMAP(n_components=NUM_OF_SEQUENCES, random_state=42, low_memory=False, verbose=True).fit(fit_data)
    return mapper.transform(sparsematrix)


def create_batch(df, start_index, end_index, cross_correlate = True):
    """This function creates a batch of a given size, translates the
    aminoacid sequence into a one-hot encoded sequence and adds the 
    two sequences together
    
    param:  df: Dataframe
            start_index: int
            end_index: int
            
    return: input data: ndarray
            labels: ndarray"""
        
    batchsize = end_index - start_index  
    labels = df["labels"].to_numpy()[start_index : end_index]
    
    if cross_correlate:
        input_data = np.empty((batchsize, len(PROTEIN_ENCODING) * len(df["start_seq"][0])**2))
        start = time.time()
        for i in range(start_index, start_index + batchsize):
            index = i % batchsize
            input_data[index,:] = add_binary_sequences(seq_into_binary(df["start_seq"][i]),
                                                    seq_into_binary(df["end_seq"][i]))
        print("Adding sequences took",time.time()-start, " seconds")
        
        # Reduce dimensionality to 20
        start = time.time()
        input_data = reduce_dimensionality(input_data)
        print("reducing took", time.time()-start,' seconds')
    
    else:
        input_data = np.empty((batchsize, len(df["start_seq"][0])*2))
        for i in range(start_index, start_index + batchsize):
            index = i % batchsize
            input_data[index,:len(df["start_seq"][0])] = df["start_seq"][i]
            input_data[index,len(df["start_seq"][0]):] = df["end_seq"][i]
        
    return input_data, labels
  
