from src.utils.preprocessing import preprocessing
from src.utils.training import train, create_model
from src.utils.utils import load_train_test_data, get_false_true_ratio_from_filename
from src.CONSTS import *
from src.utils.model_utils import transform_data


def main(): 
    
    preprocessing(force_save_seq = False, false_per_true = 1)
    
    # Try false/true ratio 1 and 4
    for file in [TRAIN_TEST_DATA_PATH_1_1, TRAIN_TEST_DATA_PATH_1_4]:
        false_per_true = get_false_true_ratio_from_filename(file)
            
        raw_data = load_train_test_data(file)
        
        # Differ modes of comparing/ non-comparing begin and end
        for operation in [ADD, MULTIPLY, CONCATENATE]:
            
            input_data, labels = transform_data(raw_data, compare = operation)
            input_size = input_data.size()[1]
            
            for model_type in [LOGISTIC_REGRESSION, NEURAL_NET_1, NEURAL_NET_2]:
                for lr in [1e-1, 1e-3, 1e-4]:
                    
                    # File name for saving performance 
                    file_name = str(false_per_true) + '_' + operation + '_' + model_type + '_' + str(lr)
                    
                    if model_type is not LOGISTIC_REGRESSION:
                        # Try different sizes for the first layer of the neural network
                        for hidden_layer_size in [50, 100, 150, 200]:
                            model = create_model(input_size, model_type, hidden_size_1 = hidden_layer_size)
                            file_name = file_name + '_' + str(hidden_layer_size) 
                            # Train model and save performance after every epoch
                            train(model, input_data, labels, model_name = file_name, lr = lr) 
                              
                    else:
                        model = create_model(input_size, model_type)
                        # Train model and save performance after every epoch
                        train(model, input_data, labels, model_name = file_name, lr = lr)


main()
