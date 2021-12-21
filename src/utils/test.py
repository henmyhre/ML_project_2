from src.utils.model_utils import *

def test(model, test_data):
    input_data, labels = transform_data(test_data)
    # Get performance after epoch
    x_batch = input_data.float()
    y_batch = labels.float()
    # x_batch = X.index_select(0, indices[-1,:]).to_dense().float()  # Get dense representation
    # y_batch = labels.index_select(0, indices[-1,:]).float()
    # get pred
    y_pred = model.forward(x_batch)
    y_pred = y_pred.reshape(y_batch.size())
    # Get metrics
    accuracy, F_score = get_performance(y_batch, y_pred)
    
    print("Accuracy is: %.4f and F1-score is: %.4f" %(accuracy, F_score))
