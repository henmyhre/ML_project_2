from preprocessing import *
from consts import *
from neural_network import *
import time

def encoding():

    unique_letters = ['V', 'W', 'H', 'G', 'T', 'Q', 'A', 'C', 'S', 'K', 'Y', 'L', 'R', 'D', 'M', 'E', 'I', 'F', 'N', 'P', '-', 'X']

    for index, letter in enumerate(unique_letters):
        PROTEIN_ENCODING[letter] = [0 for _ in range(len(unique_letters))]
        PROTEIN_ENCODING[letter][index] = 1
    
        #inverse dictionary
        BINARY_ENCODING[str(PROTEIN_ENCODING[letter])] = letter
        
        
encoding()

# Create CSV file and get size of true and false data
count_true, count_false = split_file(input_file, seq_file)
print(count_true, count_false)
#%%
start = time.time()

batch_size = 1000
shape = (count_true, count_true) # count_true = N_files

line_numbers = build_k_indices(count_true + count_false, batch_size, shape, seed = 1)  # Create batches of random entries

start = time.time()
for i in range(len(line_numbers[0][0])):
  print("time taken:", time.time() - start)
  batch = load_batch(seq_file, line_numbers[0][:,i], line_numbers[1][:,i], batch_size)
  batch_input = create_batch_input(batch)
  labels = batch["label"].astype(int).to_numpy()
  

end = time.time()
print(end - start)  
#%%


model = MLP(hidden_size=2)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
lossfunc = nn.MSELoss()


train(batch_input, labels, model, lossfunc, optimizer, 1, batch_size)
