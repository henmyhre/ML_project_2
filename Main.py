from src.utils.preprocessing import preprocessing
from src.utils.training import train, create_model
from src.utils.utils import load_train_test_data
from src.utils.test import test
from src.CONSTS import MODEL_TYPE, LR, FILES, TRAIN_TEST_DATA_PATH_1_1
from src.utils.model_utils import transform_data
import torch


def main(): 
    
    
    preprocessing(force_save_seq = True, false_per_true = 1)
    
    for file in FILES:
        if file == TRAIN_TEST_DATA_PATH_1_1:
            false_per_true = 1
        else:
            false_per_true = 4
            
        raw_data = load_train_test_data(file)
        # Set compare to False to not compare start and end sequence by multiplication
        input_data, labels = transform_data(raw_data, compare = True)
        
        for model_type in MODEL_TYPE:
            for lr in LR:
              for number in [50, 100, 150, 200]:
                model = create_model(input_data.size()[1], net = model_type, hidden_size_1=number)
                model_type = model_type + ", " + str(number)
                train(model, input_data, labels, false_per_true, model_name = model_type, lr = lr)
        
    return model

model = main()
