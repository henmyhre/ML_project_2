from Bio import SeqIO
from src.CONSTS import * 
from src.utils.utils import *
import time
 

def create_encoding_dictionaries(): 
  """"
  Creates protein encoding dictionaries for hot-key encoding.
  The dictionaries are saved in CONSTS.py.
  """
  for index, letter in enumerate(UNIQUE_CHARS):
    PROTEIN_ENCODING[letter] = [0 for _ in range(len(UNIQUE_CHARS))]
    PROTEIN_ENCODING[letter][index] = 1

    # Inverse dictionary
    BINARY_ENCODING[str(PROTEIN_ENCODING[letter])] = letter
   
    

def split_raw_data_file():
    """
    Splits the raw data file and saves true and false combinations of sequences in separate files.
    """
    
    fasta_sequences = SeqIO.parse(open(RAW_DATA_FILE_PATH),'fasta')
    
    seq_start_list = []
    seq_end_list = []
    
    for fasta in fasta_sequences:
        name, sequence = fasta.id, str(fasta.seq)
        sequence_start, sequence_end = split_sequence(sequence)
        seq_start_list.append([name, sequence_start])
        seq_end_list.append([name, sequence_end])
    
    with open(SEQ_TRUE_FILE_PATH, "w+") as seq_true_file, open(SEQ_FALSE_FILE_PATH, "w+") as seq_false_file: 
        
        for start in seq_start_list:   
            for end in seq_end_list:
                name_and_seq = start[0] + "<>" + end[0] + ";" + start[1] + ";" + end[1]
                label = "-1"
                
                if start[0] == end[0]: 
                    label = "1"
                seq_true_file.write(name_and_seq + ";" + label + "\n")


        seq_true_file.close()
        seq_false_file.close()
  

def create_train_test_data(false_per_true):
    """
    Saves all true data and a random sample of the false data to a file for training and testing.
    """
    
    true_amount = NUM_OF_SEQUENCES
    false_amount = NUM_OF_SEQUENCES*false_per_true
    
    train_test_sample_file_path = TRAIN_TEST_DATA_PATH_BASE + "1_" + str(false_per_true) + ".csv"
    
    clear_file(train_test_sample_file_path)
    
    append_csv_rows_to_new_csv(SEQ_TRUE_FILE_PATH, train_test_sample_file_path, true_amount)
    append_csv_rows_to_new_csv(SEQ_FALSE_FILE_PATH, train_test_sample_file_path, false_amount)
    
    return train_test_sample_file_path


def preprocessing(force_save_seq = False, false_per_true = 1):
    """
    From raw data file to file with random train/test data in random order.
    """
    if (len(BINARY_ENCODING) == 0) or (len(PROTEIN_ENCODING) == 0):  
        # Create dictionaries for encoding proteins
        create_encoding_dictionaries()
        
    start = time.time()
    
    if force_save_seq:        
        print(get_time_dif_str(start), "Starting saving to files")
        
        split_raw_data_file()
        print(get_time_dif_str(start), "Done saving to files true false files. Shuffling...")
        
        shuffle_file_rows(SEQ_FALSE_FILE_PATH)
        print(get_time_dif_str(start), "Done shuffling. Saving preprocessed data...")
    
    else:
        print("Skipped creating true/false files. Saving preprocessed data...")
        
    train_test_data_file = create_train_test_data(false_per_true)
    print(get_time_dif_str(start), "Done saving preprocessed data. Shuffling...")
    
    shuffle_file_rows(train_test_data_file)
    print(get_time_dif_str(start), "Done with shuffling. \n Done with preprocessing!:)")
    
    

  

