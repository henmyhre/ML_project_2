from preprocessing import *
from consts import *
from neural_network import *
import time  
import os     

def main(): 
  
  if len(BINARY_ENCODING) == 0 or len(PROTEIN_ENCODING) == 0:  
    # Create dictionaries for encoding proteins
    create_encoding_dictionaries()

  # Create CSV file and get size of true and false data
  if len(os.listdir(preprocessed_seq_path)) == 0:
    count_true, count_false = split_file(raw_data_path, preprocessed_seq_path)
  
  print(count_true, count_false)

  start = time.time()

  batch_size = 1000
  shape = (count_true, count_true) # count_true = N_files

  line_numbers = build_k_indices(count_true + count_false, batch_size, shape, seed = 1)  # Create batches of random entries

  start = time.time()
  for i in range(len(line_numbers[0][0])):
    print("time taken:", time.time() - start)
    batch = load_batch(preprocessed_seq_path, line_numbers[0][:,i], line_numbers[1][:,i], batch_size)
    batch_input = create_batch_input(batch)
    labels = batch["label"].astype(int).to_numpy()
    

  end = time.time()
  print(end - start)  

  model = MLP(hidden_size=2)
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
  lossfunc = nn.MSELoss()

  train(batch_input, labels, model, lossfunc, optimizer, 1, batch_size)

main()
