from src.utils.preprocessing import preprocessing
from src.utils.training import train, create_model
from src.utils.utils import load_train_test_data, show_performance
from src.CONSTS import MODEL_TYPE, LR, FILES, TRAIN_TEST_DATA_PATH_1_1, LOGISTIC_REGRESSION, COMPARE
from src.utils.model_utils import transform_data
import numpy as np

def main(): 
    
    # Get data from fasta file into csv and create false examples
    preprocessing(force_save_seq = True, false_per_true = 1)
    
    optim_f = np.zeros((len(FILES), len(COMPARE), len(MODEL_TYPE), len(LR), 3))
    optim_acc = np.zeros((len(FILES), len(COMPARE), len(MODEL_TYPE), len(LR), 3)  )                 
    # Try false/true ratio one and false/true ratio 4
    for i_file, file in enumerate(FILES):
        if file == TRAIN_TEST_DATA_PATH_1_1:
            false_per_true = 1
        else:
            false_per_true = 4
            
        raw_data = load_train_test_data(file)
        
        # Differ modes of comparing/ non-comparing begin and end (add, multiply, concatenate)
        for i_comp, comparison in enumerate(COMPARE):
            
            input_data, labels = transform_data(raw_data, compare = comparison)
            # Try different models: logistic regression, one layer neural network and two layer neural network
            for i_mod, model_type in enumerate(MODEL_TYPE):
                for i_lr, lr in enumerate(LR):
                    if model_type is not LOGISTIC_REGRESSION:
                        # Try different sizes for the first layer of the neural network
                        for i_num, number in enumerate([50, 100, 200]):
                            # Create model
                            model = create_model(input_data.size()[1], model_type = model_type, hidden_size_1=number)
                            # Set file name to save performance
                            file_name = str(false_per_true) + '_' + comparison + '_' + model_type + "_" + str(number) + '_' + str(lr)
                            # Train model and safe performance after every epoch
                            f_score, accuracy = train(model, input_data, labels, model_name = file_name, lr = lr)
                            
                            optim_f[i_file, i_comp, i_mod, i_lr, i_num] = f_score
                            optim_acc[i_file, i_comp, i_mod, i_lr, i_num] = accuracy
                    else:
                        # Create model
                        model = create_model(input_data.size()[1], model_type = model_type)
                        # Set file name to save performance
                        file_name = str(false_per_true) + '_' + comparison + '_' + model_type + '_' + str(lr)
                        # Train model and safe performance after every epoch
                        f_score, accuracy = train(model, input_data, labels, model_name = file_name, lr = lr)
                        
                        optim_f[i_file, i_comp, i_mod, i_lr, 0] = f_score
                        optim_acc[i_file, i_comp, i_mod,i_lr, 0] = accuracy
                        
    show_performance(optim_f,  optim_acc)                    
                        

main()


