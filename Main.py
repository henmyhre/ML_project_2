from preprocessing import *
from consts import *
from neural_network import *
import time
import sh

def encoding():

    unique_letters = ['V', 'W', 'H', 'G', 'T', 'Q', 'A', 'C', 'S', 'K', 'Y', 'L', 'R', 'D', 'M', 'E', 'I', 'F', 'N', 'P', '-']

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
line_numbers = build_k_indices(count_true + count_false, batch_size, seed = 2)  # Create batches of random entries

for serie in line_numbers:
  print("time taken:", time.time() - start)
  batch = load_batch(seq_file, serie)
  batch_input = create_batch_input(batch)
  labels = batch["label"].astype(int).to_numpy()
  sh.shuf("words.txt", out="shuffled_words.txt")

end = time.time()
print(end - start)  
#%%


model = MLP(hidden_size=2)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
lossfunc = nn.MSELoss()


train(batch_input, labels, model, lossfunc, optimizer, 1, batch_size)
