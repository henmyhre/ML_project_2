from src.utils.preprocessing import preprocessing
from src.utils.training import train, create_model
from src.utils.utils import load_train_test_data
from src.CONSTS import MODEL_TYPE, LR, FILES, TRAIN_TEST_DATA_PATH_1_1, LOGISTIC_REGRESSION, COMPARE
from src.utils.model_utils import transform_data


def main(): 
    
    # Get data from fasta file into csv and create false examples
    preprocessing(force_save_seq = True, false_per_true = 1)
    
    # Try false/true ratio one and false/true ratio 4
    for file in FILES:
        if file == TRAIN_TEST_DATA_PATH_1_1:
            false_per_true = 1
        else:
            false_per_true = 4
            
        raw_data = load_train_test_data(file)
        
        # Differ modes of comparing/ non-comparing begin and end (add, multiply, concatenate)
        for comparison in COMPARE:
            
            input_data, labels = transform_data(raw_data, compare = comparison)
            # Try different models: logistic regression, one layer neural network and two layer neural network
            for model_type in MODEL_TYPE:
                for lr in LR:
                    if model_type is not LOGISTIC_REGRESSION:
                        # Try different sizes for the first layer of the neural network
                        for number in [50, 100, 150, 200]:
                            # Create model
                            model = create_model(input_data.size()[1], model_type = model_type, hidden_size_1=number)
                            # Set file name to save performance
                            file_name = str(false_per_true) + '_' + comparison + '_' + model_type + "_" + str(number) + '_' + str(lr)
                            # Train model and safe performance after every epoch
                            train(model, input_data, labels, model_name = file_name, lr = lr)
                            
                    else:
                        # Create model
                        model = create_model(input_data.size()[1], model_type = model_type)
                        # Set file name to save performance
                        file_name = str(false_per_true) + '_' + comparison + '_' + model_type + '_' + str(lr)
                        # Train model and safe performance after every epoch
                        train(model, input_data, labels, model_name = file_name, lr = lr)
                        
    return model

model = main()
