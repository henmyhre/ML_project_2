from preprocessing import *
from consts import *
from neural_network import *
import time  
import os     


def preprocessing(force_save_seq):
  if (len(BINARY_ENCODING) == 0) or (len(PROTEIN_ENCODING) == 0):  
    # Create dictionaries for encoding proteins
    create_encoding_dictionaries()
    
  # Create CSV files and get size of true and false data
  if (len(os.listdir(preprocessed_seq_path)) != 5353) or force_save_seq:
    print("Starting saving to files")
    count_true, count_false = split_file(raw_data_file_path, seq_file_path)
    
    with open(num_true_false_file_path, "w") as num_true_false:
      num_true_false.write(str(count_true) + ";" + str(count_false))
      
    print("Done saving to files. " + "True count: " + str(count_true) + "; False count: " + str(count_false))
    
  else:
    f = open(num_true_false_file_path)
    csv_reader = csv.reader(f)
    counts = next(csv_reader)
    
    count_true, count_false = [int(count) for count in counts[0].split(";")]
    print("Skipped saving to files. " + "True count: " + str(count_true) + "; False count: " + str(count_false))

  return count_true, count_false


def create_model():
  model = MLP(hidden_size=2, lossfunc=nn.MSELoss())
  model.set_optimizer()
  
  return model


def train_model(count_true, count_false, model):
  start = time.time()

  batch_size = 1000
  shape = (count_true, count_true) # count_true = N_files
  
  line_numbers = build_k_indices(count_true + count_false, batch_size, shape, seed = 1)  # Create batches of random entries

  for i in range(len(line_numbers[0][0])):
    print(str(i) + "th iteration. Time taken:", time.time() - start)
    batch = load_batch(seq_file_path, line_numbers[0][:,i], line_numbers[1][:,i], batch_size)
    batch_input = create_batch_input(batch)
    labels = batch["label"].astype(int).to_numpy()

  end = time.time()
  print("Training time: " + str(end - start))

  train(batch_input, labels, model, model.lossfunc, model.optimizer, 1, batch_size)


def main(force_save_seq = False): 
  
  count_true, count_false = preprocessing(force_save_seq)
  
  model = create_model(count_true, count_false)
  
  train_model(count_true, count_false, model)
    


main()
