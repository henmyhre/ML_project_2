from src.utils.preprocessing import preprocessing
from src.utils.training import train, create_model
from src.utils.utils import load_train_test_data, get_false_true_ratio_from_filename
from src.CONSTS import *
from src.utils.model_utils import transform_data, show_performance
import numpy as np

def main(): 
    
    preprocessing(force_save_seq = False, false_per_true = 1)
    
    optim_f = np.zeros((2, 3, 3, 3, 3))
    optim_acc = np.zeros((2, 3, 3, 3, 3))  
               
    # Try false/true ratio one and false/true ratio 4
    for i_file, file in enumerate(FILES):

        false_per_true = get_false_true_ratio_from_filename(file)  
        raw_data = load_train_test_data(file)
        
        # Differ modes of comparing/ non-comparing begin and end (add, multiply, concatenate)
        for i_comp, operation in enumerate(OPERATIONS):
            
            input_data, labels = transform_data(raw_data, operation = operation)
            input_size = input_data.size()[1]
            
            # Try different models: logistic regression, one layer neural network and two layer neural network
            for i_mod, model_type in enumerate(MODEL_TYPES):
                for i_lr, lr in enumerate(LEARNING_RATES):
                  
                    # File name for saving performance 
                    file_name = str(false_per_true) + '_' + operation + '_' + model_type + '_' + str(lr)
                    
                    if model_type is not LOGISTIC_REGRESSION:
                        # Try different sizes for the first layer of the neural network
                        for i_size, hidden_layer_size in enumerate([50, 100, 200]):
                            model = create_model(input_size, model_type = model_type, hidden_size_1=hidden_layer_size)
                            file_name = file_name + '_' + str(hidden_layer_size) 
                            # Train model and safe performance after every epoch
                            f_score, accuracy = train(model, input_data, labels, model_name = file_name, lr = lr)
                            
                            optim_f[i_file, i_comp, i_mod, i_lr, i_size] = f_score
                            optim_acc[i_file, i_comp, i_mod, i_lr, i_size] = accuracy
                            
                    else:
                        model = create_model(input_size, model_type = model_type)
                        # Train model and safe performance after every epoch
                        f_score, accuracy = train(model, input_data, labels, model_name = file_name, lr = lr)
                        optim_f[i_file, i_comp, i_mod, i_lr, i_size] = f_score
                        optim_acc[i_file, i_comp, i_mod, i_lr, i_size] = accuracy
                            
    show_performance(optim_f, optim_acc)    
                
main()


   
