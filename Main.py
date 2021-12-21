from src.utils.preprocessing import preprocessing
from src.utils.training import train, create_model
from src.utils.utils import load_train_test_data
from src.utils.test import test
from src.utils.model_utils import transform_data
import torch


def main(): 
    preprocessing(force_save_seq = False, false_per_true = 1)
    
    raw_data = load_train_test_data()
    input_data, labels = transform_data(raw_data)
    
    
    model = create_model(input_data.size()[1])
    train(model, input_data, labels)
    
    return model

model = main()
