from typing import Tuple
from src.utils.preprocessing import preprocessing
from src.utils.training import create_model, train_model
from src.CONSTS import *


def main(): 
  preprocessing(force_save_seq = True)
  model = create_model()
  train_model(model)

main()
