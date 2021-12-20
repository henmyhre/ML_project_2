# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 09:33:54 2021

@author: jurri
"""


import numpy as np
import scipy.sparse
import sklearn.datasets
import umap
from src.utils.make_batch import create_sparse_matrix_scipy, create_sparse_matrix_pytorch
from src.CONSTS import train_test_sample_file_path
from src.utils.preprocessing import create_encoding_dictionaries
import pandas as pd
import time
def load_data():
  data = pd.read_csv(train_test_sample_file_path,
                     names = ["name","start_seq", "end_seq", "labels"], sep=';')
  return data

create_encoding_dictionaries()

df =load_data()
#%%
from src.utils.make_batch import create_sparse_matrix_scipy, create_sparse_matrix_pytorch

sparse = create_sparse_matrix_pytorch(df[:20])

#%%
import torch
import numpy as np
def build_indices_batches(num_row, interval, seed):
    """build k indices for k-fold."""
    
    k_fold = int(num_row / interval)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return torch.tensor(k_indices)
  
  
a = build_indices_batches(20, 2, 2)


#%%
from sklearn.decomposition import TruncatedSVD
start = time.time()
print("Start...")
sparse = create_sparse_matrix_scipy(df)
#%%
from sklearn.decomposition import IncrementalPCA

start = time.time()
print("Start...")
n_size = 300

pca = IncrementalPCA(n_components = n_size, batch_size = 500)
pca.fit(sparse)
print(pca.explained_variance_ratio_.sum())
print(time.time()-start, "fitting time")

output = np.zeros((list(sparse.shape)[0], n_size))
chunks = np.linspace(0, list(sparse.shape)[0], 20, dtype = int)
for i in range(len(chunks)-1):
    sub_set = sparse[chunks[i]:chunks[i+1], :]
    output[chunks[i]:chunks[i+1], :] = pca.transform(sub_set)
    
print(time.time()-start, "total time")


# =============================================================================
# for comp in range(250, df3.shape[1]):
#     print("doing pca ...")
#     pca = TruncatedSVD(n_components= comp, random_state=42)
#     pca.fit(df3)
#     comp_check = pca.explained_variance_ratio_
#     final_comp = comp
#     print(comp_check.sum())
#     if comp_check.sum() > 0.85:
#         break
#         
# Final_PCA = TruncatedSVD(n_components= final_comp,random_state=42)
# Final_PCA.fit(df3)
# cluster_df=Final_PCA.transform(df3)
# num_comps = comp_check.shape[0]
# =============================================================================




#print("Finished in", time.time()-start, "seconds")
#%%
def reduce_dimensionality(sparsematrix):
    """This function reduces the dimension of the sparsematrix"""
    # Fit reduce model on 10% of data
    size = int(list(sparsematrix.shape)[0]*0.05)
    fit_data = sparsematrix[:size, :]
    print("train reduce ...")
    mapper = umap.UMAP(n_neighbors = 15, 
                       n_components=20,
                       random_state=42,
                       low_memory=True,
                       verbose=True).fit(fit_data)
    
    output = np.zeros((list(sparsematrix.shape)[0], 20))
    for i in range(20):
        output = np.zeros((list(sparsematrix.shape)[0], 20))
        mapper.transform(sparsematrix)
        
        
    
    return 
  
# = time.time()
 
a = reduce_dimensionality(sparse)

print("Finished in", time.time()-start, "seconds")