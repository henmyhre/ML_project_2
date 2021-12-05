from typing import Tuple
from src.utils.preprocessing import preprocessing
from src.utils.training import create_model, train_model


def main(): 
  preprocessing(force_save_seq = True, false_per_true = 1)
  model = create_model()
  train_model(model)

main()
