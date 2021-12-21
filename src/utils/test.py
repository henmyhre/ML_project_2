from src.utils.model_utils import *

def test(model, test_data):
    input_data, labels = transform_data(test_data)
    # Get performance after epoch
    x_test = input_data.float()
    y_test = labels.float()
    # get pred
    y_pred = model.forward(x_test)
    y_pred = y_pred.reshape(y_test.size())
    # Get metrics
    accuracy, F_score = get_performance(y_test, y_pred)
    
    print("Accuracy is: %.4f and F1-score is: %.4f" %(accuracy, F_score))
