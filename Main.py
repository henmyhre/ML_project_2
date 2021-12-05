from src.utils.preprocessing import *
from src.CONSTS import *
from src.neural_network import *
from src.utils.training import *


def main(force_save_seq = False): 
  preprocessing(force_save_seq)
  model = create_model()
  train_model(model)

main()
