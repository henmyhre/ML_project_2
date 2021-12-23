import random 
import time
import pandas as pd
from src.CONSTS import *


def clear_file(file_path):
    """
    Clears file by overwriting with nothing.
    param: flie_path: str
    """
    f = open(file_path, "w+")
    f.close()


def read_lines(file_path):
    """
    Read all lines of a file.
    param: file_path: str
    return: list
    """
    return open(file_path).readlines()


def write_lines(file_path, lines):
    """
    Clear file and write new lines to file .
    param: file_path: str
           lines: list
    """
    open(file_path, 'w').writelines(lines)


def append_lines(file_path, lines):
    """
    Append lines to existing file.
    param: file_path: str
           lines: list
    """
    open(file_path, 'a').writelines(lines)


def shuffle_list(l):
    """
    Shuffles list.
    param: l: list
    """
    return random.shuffle(l)


def shuffle_file_rows(file_path):
    """
    Shuffle rows in file
    param: file_path: str
    """
    lines = read_lines(file_path)
    shuffle_list(lines)
    write_lines(file_path, lines)


def append_file_rows_to_new_file(src_path, dst_path, amount):
    """
    Extracts rows from src file, randomize the order and add them to dst file.
    param: src_path: str
           dst_path: str
           amount: int
    """
    lines = read_lines(src_path)[:amount-1]
    shuffle_list(lines)
    append_lines(dst_path, lines)


def split_sequence(sequence):
    """
    Splits a sequence in the middle.
    param: sequence: str
    return: start: str
            end: str
    """
    seq_len = len(sequence)
    half_index = int(seq_len/2)
    start, end = sequence[:half_index], sequence[half_index:]
    return start, end


def get_time_dif_str(start):
    """
    Returns string to show time difference from provided start time.
    param: start: float
    return: str
    """
    return "Time: " + str(int(time.time() - start))


def load_train_test_data(file_path):
    """
    Load a file into a dataframe.
    param: file_path: str
    return: data: pd.DataFrame
    """
    
    data = pd.read_csv(file_path,
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



def get_false_true_ratio_from_filename(filename):
    """
    Extracts the last number in the file name since this is the ratio of false/true
    param: filename: str
    return: int
    """
    return int(filename.split(".")[0][-1])

  
  
  
  
  
  
  
  
  
