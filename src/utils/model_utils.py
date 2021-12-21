from sklearn.metrics import accuracy_score, f1_score
import torch
import numpy as np
from src.CONSTS import PROTEIN_ENCODING
from sklearn.decomposition import IncrementalPCA



def transform_data(df, compare = True):
    """
    This function transforms the raw data into input data for the learning algorithm
    and into the corrseponding labels.
    param: df: pd.DataFrame
    
    return: torch.tensor
            torch.tensor
    """
    
    begin, end, labels = create_one_hot_encoded(df)
    
    begin_reduced = pca_transform(begin, n = 400)
    end_reduced = pca_transform(end, n = 400)
    
    if compare:
        # Compare by mulitplication  
        return torch.tensor(begin_reduced*end_reduced), torch.tensor(labels)
    
    begin_end_reduced = np.concatenate((begin_reduced, end_reduced), axis=1)
    return torch.tensor(begin_end_reduced), torch.tensor(labels)
      
      
 
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



def create_one_hot_encoded(df):
    """
    This function "translates" the amino acid sequences into one hot encoded sequences.
    param: df: pd.DataFrame, contains startand end sequence in columns start_seq and end_seq
    
    return: begin_one_hot: ndarray
            end_one_hot: ndarray
            labels: list
    """
    
    begin_one_hot = np.empty((len(df), len(PROTEIN_ENCODING)*len(df["start_seq"].iloc[0])))
    end_one_hot = np.empty((len(df), len(PROTEIN_ENCODING)*len(df["end_seq"].iloc[0])))
    
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
    """
    This functions does pca on the input data. PCA finds n principle components.
    param: data: ndarray
           n: int
    return: ndarray
    """
    pca = IncrementalPCA(n_components = n, batch_size=500)
    pca.fit(data)
    print("Variance explained:",pca.explained_variance_ratio_.sum())
    return pca.transform(data)
   
    
    
    

def get_performance(y_true, y_pred):
        
    y_true = y_true.numpy()
    
    y_pred = y_pred.detach()
    sig = torch.nn.Sigmoid()
    y_pred = sig(y_pred)
    y_pred = np.round(y_pred.numpy())
    accuracy = accuracy_score(y_true, y_pred)
    F_score = f1_score(y_true, y_pred)
    return accuracy, F_score
