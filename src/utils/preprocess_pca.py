# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 16:44:00 2021

@author: jurri
"""


import numpy as np
from src.CONSTS import PROTEIN_ENCODING
import torch
from sklearn.decomposition import IncrementalPCA

 
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
  
# =============================================================================
# create_encoding_dictionaries()
# 
# df = load_data()
# =============================================================================


def create_one_hot_encoded(df):
    begin_one_hot = np.empty((len(df), len(PROTEIN_ENCODING)*len(df["start_seq"][0])))
    end_one_hot = np.empty((len(df), len(PROTEIN_ENCODING)*len(df["end_seq"][0])))
    
    df["labels"] = df["labels"].replace(-1, 0)
    labels = list()
    
    index = 0
    while not df.empty:   
        row = df.iloc[-1]   # Read last row
        df = df.iloc[:-1]   # Delete last row to save memory
        # One hot encode amino acid sequences and put both ends into an array
        begin_one_hot[index, :] = seq_into_binary(row["start_seq"]).flatten()
        end_one_hot[index, :] = seq_into_binary(row["end_seq"]).flatten()
        # Also save labels
        labels.append(row["labels"].item())
        index +=1
    return begin_one_hot, end_one_hot, labels
  
def pca_transform(data, n = 300):
    pca = IncrementalPCA(n_components = n, batch_size=500)
    pca.fit(data)
    print("Variance explained:",pca.explained_variance_ratio_.sum())
    
    return pca.transform(data)


def transform_data(df):
    
    begin, end, labels = create_one_hot_encoded(df)
    begin_reduced = pca_transform(begin, n = 400)
    end_reduced= pca_transform(end, n = 400)
    # Compare by mulitplication
    return torch.tensor(begin_reduced*end_reduced), torch.tensor(labels)
    
