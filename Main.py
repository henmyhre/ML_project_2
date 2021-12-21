from src.utils.preprocessing import preprocessing
from src.utils.training import train
from src.utils.utils import load_train_test_data
from src.utils.test import test
from src.utils.model_utils import transform_data


def main(): 
    preprocessing(force_save_seq = False, false_per_true = 1)
    
    raw_data = load_train_test_data()
    train_data=raw_data.sample(frac=0.8,random_state=200)
    test_data=raw_data.drop(train_data.index)
    
    train_input_data, train_labels = transform_data(train_data)
    test_input_data, test_labels = transform_data(test_data)
    
    train(model, train_input_data, train_labels)
    test(model, test_input_data, test_labels)
    
    return model

model = main()
