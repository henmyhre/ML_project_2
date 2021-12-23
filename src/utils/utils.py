import random 
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from src.CONSTS import *


def clear_file(file):
    f = open(file, "w+")
    f.close()

def read_lines(file):
    return open(file).readlines()

def write_lines(file, lines):
    open(file, 'w').writelines(lines)
  
def append_lines(file, lines):
    open(file, 'a').writelines(lines)

def shuffle_lines(lines):
    return random.shuffle(lines)

def shuffle_file_rows(file):
    lines = read_lines(file)
    shuffle_lines(lines)
    write_lines(file, lines)

def append_csv_rows_to_new_csv(src, dst, amount):
    lines = read_lines(src)[:amount-1]
    shuffle_lines(lines)
    append_lines(dst, lines)

def split_sequence(sequence):
    seq_len = len(sequence)
    start, end = sequence[:int(seq_len/2)], sequence[int(seq_len/2):]
    return start, end

def get_time_dif_str(start):
    return "Time: " + str(int(time.time() - start))

def load_train_test_data(file_name):
    """Load a file into a dataframe.
    param: file_name: str
    return: data: pd.DataFrame"""
    
    data = pd.read_csv(file_name,
                        names = ["name","start_seq", "end_seq", "labels"], sep=';')
    return data
  
  
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
   # plt.title('For learning rate =' + str(lr) + ', model is ' + model_name + ". False/true ratio is:" + str(false_per_true))
    plt.savefig('generated/'+ model_name + '_' + y_label + '.png', bbox_inches = 'tight')
    
    plt.show()
  
def show_performance(optim_f, optim_acc):
    """This function shows the performance of various training configurations.
    param: optim_f: ndarray
           optim_acc: ndarray"""
    
    
    shape = (len(FILES), len(COMPARE), len(MODEL_TYPE), len(LR), 3)
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

  
  
  
  
  
  
  
  
