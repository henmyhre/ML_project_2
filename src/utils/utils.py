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
    data = pd.read_csv(file_name,
                        names = ["name","start_seq", "end_seq", "labels"], sep=';')
    return data
  
  
def plot_result(data, x_label, y_label, lr, model_name, false_per_true):
    plt.plot(np.array(data))
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title('For learning rate =' + str(lr) + ', model is ' + model_name + ". False/true ratio is:" + str(false_per_true))
    plt.savefig('generated/'+y_label+'_'+str(lr)+'_'+model_name+'_'+str(false_per_true)+'.png', bbox_inches = 'tight')
    
    plt.show()
  
  
  
  
  
  
  
  
  
  
