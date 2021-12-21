

def test(model, test_data):
    # Get performance after epoch
    x_batch = X[indices[-1,:]].float()
    y_batch = labels[indices[-1,:]].float()
    # x_batch = X.index_select(0, indices[-1,:]).to_dense().float()  # Get dense representation
    # y_batch = labels.index_select(0, indices[-1,:]).float()
    # get pred
    y_pred = model.forward(x_batch)
    y_pred = y_pred.reshape(y_batch.size())
    # Get metrics
    accuracy, F_score = get_performance(y_batch, y_pred)
    
    
    print("Epoch ",k," finished, total time taken:", time.time()-start)
    print("Accuracy is: %.4f and F1-score is: %.4f" %(accuracy, F_score))
