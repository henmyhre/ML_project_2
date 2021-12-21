from src.utils.preprocessing import preprocessing
from src.utils.training import train
from src.utils.utils import load_data


def main(): 
    preprocessing(force_save_seq = False, false_per_true = 1)
    
    raw_data = load_data()
    train_data=raw_data.sample(frac=0.8,random_state=200) #random state is a seed value
    test_data=raw_data.drop(train_data.index)
    
    model = train(train_data)
    return model

model = main()
