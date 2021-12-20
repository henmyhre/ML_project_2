from src.utils.preprocessing import preprocessing
from src.utils.training import train


def main(): 
  preprocessing(force_save_seq = False, false_per_true = 1)
  model = train()  
  return model

model = main()
