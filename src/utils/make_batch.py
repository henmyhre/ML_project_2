# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 08:22:05 2021

@author: jurri
"""
from Bio import SeqIO
from src.CONSTS import * 
import pandas as pd
import numpy as np
from src.CONSTS import *


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
    for i in range(start_index, start_index + batchsize):
      index = i % batchsize
      input_data[index,:] = add_binary_sequences(seq_into_binary(df["start_seq"][i]),
                                             seq_into_binary(df["end_seq"][i]))
      
  
  else:
    input_data = np.empty((batchsize, len(df["start_seq"][0])*2))
    for i in range(start_index, start_index + batchsize):
      index = i % batchsize
      input_data[index,:len(df["start_seq"][0])] = df["start_seq"][i]
      input_data[index,len(df["start_seq"][0]):] = df["end_seq"][i]
    
  return input_data, labels
  
  
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