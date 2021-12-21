from src.utils.preprocessing import preprocessing
from src.utils.training import train
from src.utils.utils import load_train_test_data
from src.utils.test import test
from src.utils.model_utils import transform_data
import torch


def main(): 
    preprocessing(force_save_seq = False, false_per_true = 1)
    
    raw_data = load_train_test_data()
    input_data, labels = transform_data(raw_data)
    
    train_input_data, test_input_data = torch.split(input_data, input_data.size()[0]*0.8)
    train_labels, test_labels = torch.split(labels, labels.size()[0]*0.8)
    
    
    train(model, train_input_data, train_labels)
    test(model, test_input_data, test_labels)
    
    return model

model = main()
