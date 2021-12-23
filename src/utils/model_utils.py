from sklearn.metrics import accuracy_score, f1_score
import torch
import numpy as np
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
from src.CONSTS import *



def transform_data(df, operation = None):
    """
    This function transforms the raw data into input data for the learning algorithm
    and into the corrseponding labels.
    param: df: pd.DataFrame
           operation: Type of operation between start and end sequences
    
    return: torch.tensor
            torch.tensor
    """
    
    begin, end, labels = create_one_hot_encoded(df)
    
    begin_reduced = pca_transform(begin, n = 300)
    end_reduced = pca_transform(end, n = 300)
    
    labels_tensor = torch.tensor(labels)
    
    if operation == ADD:
        # Compare by mulitplication  
        return torch.tensor(begin_reduced+end_reduced), labels_tensor
    
    elif operation == MULTIPLY:
        # Compare by mulitplication  
        return torch.tensor(begin_reduced*end_reduced), labels_tensor
    
    begin_end_reduced = np.concatenate((begin_reduced, end_reduced), axis=1)
    return torch.tensor(begin_end_reduced), labels_tensor
      
      
 
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
   

def test(model, input_data, labels):
    """"
    This function tests a given model with input_data and labels and returns accuracy and F1-score
    param: model: Module
           input_data: torch.tensor
           labels: torch.tensor
    return: accuracy: float
            F_score: float
    """
    
    # Ensure data is right type
    x_test = input_data.float()
    y_test = labels.float()
    # get pred
    y_pred = model.forward(x_test)
    y_pred = y_pred.reshape(y_test.size())
    # Get metrics
    accuracy, F_score = get_performance(y_test, y_pred)
        
    return accuracy, F_score
    

def get_performance(y_true, y_pred):
    """
    This fucntion gets the performance of a prediction.
    param: y_true: torch.tensor, true labels
           y_pred: torch.tensor, predicted labels
    return: accuracy: float
            F_score: float
    """
    # Change type, so it will properly work in the sklearn function of accuracy_score and f1_score
    y_true = y_true.numpy()
    
    # Get labels of the prediction
    y_pred = y_pred.detach()
    sig = torch.nn.Sigmoid()
    y_pred = sig(y_pred)
    y_pred = np.round(y_pred.numpy())
    
    # Get performance
    accuracy = accuracy_score(y_true, y_pred)
    F_score = f1_score(y_true, y_pred)
    return accuracy, F_score


def plot_result(data, x_label, y_label, model_name):
    """
    Plot the input data and safe under the concatenation of model_name and y_label.
    param: data: list
           x_label: str
           y_label: str
           model_name: str
    """
    plt.plot(np.array(data))
    hfont = {'fontname':'Helvetica'}
    plt.xlabel(x_label, **hfont)
    plt.ylabel(y_label, **hfont)
    #plt.title('For learning rate =' + str(lr) + ', model is ' + model_name + ". False/true ratio is:" + str(false_per_true))
    plt.savefig('generated/'+ model_name + '_' + y_label + '.png', bbox_inches = 'tight')
    
    plt.show()


def show_performance(optim_f, optim_acc):
    """
    This function shows the performance of various training configurations.
    param: optim_f: ndarray
           optim_acc: ndarray
    """
    
    shape = (len(FILES), len(OPERATIONS), len(MODEL_TYPES), len(LEARNING_RATES), 3)
    # Find best performance logistic regression
    index = np.unravel_index(optim_f[:, :, 0,:, :].argmax(), shape)
    print("The best performance for logistic regression was accuracy of %.4f and F-score of %.4f" %(optim_acc[index], optim_f[index]))
    print("Coordinates:", index)
    # Find best performance concatenating logistci regression
    index = np.unravel_index(optim_f[:, 2, 0, :, :].argmax(), shape)
    print("The best performance for logistic regression with concatenating sequences was accuracy of %.4f and F-score of %.4f" %(optim_acc[index], optim_f[index]))
    print("Coordinates:", index)
    # Find best overall performance
    index = np.unravel_index(optim_f.argmax(), shape)
    print("The best performance overall was accuracy of %.4f and F-score of %.4f" %(optim_acc[index], optim_f[index]))
    print("Coordinates:", index)
    # Find best One layer performance
    index = np.unravel_index(optim_f[:, :, 1,:,:].argmax(), shape)
    print("The best performance One layer was accuracy of %.4f and F-score of %.4f" %(optim_acc[index], optim_f[index]))
    print("Coordinates:", index)
    # Find best adding performance
    index = np.unravel_index(optim_f[:,0,:,:,:].argmax(), shape)
    print("The best adding performance  was accuracy of %.4f and F-score of %.4f" %(optim_acc[index], optim_f[index]))
    print("Coordinates:", index)
    # Find best multiplying performance
    index = np.unravel_index(optim_f[:,1,:,:,:].argmax(), shape)
    print("The est multiplying performance was accuracy of %.4f and F-score of %.4f" %(optim_acc[index], optim_f[index]))
    print("Coordinates:", index)
    #Find best 1to1 performance
    index = np.unravel_index(optim_f[0,:,:,:,:].argmax(), shape)
    print("The best 1to1 performance was accuracy of %.4f and F-score of %.4f" %(optim_acc[index], optim_f[index]))
    print("Coordinates:", index)
    #find best 4to1 performance
    index = np.unravel_index(optim_f[1,:,:,:,:].argmax(), shape)
    print("The best 4to1 performance was accuracy of %.4f and F-score of %.4f" %(optim_acc[index], optim_f[index]))
    print("Coordinates:", index)
