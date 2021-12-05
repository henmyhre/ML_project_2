
raw_data_file_path = "data/NoGapsMSA_SIS1-on-top.fasta"

seq_true_file_path = "data/sequences_true.csv"
seq_false_file_path = "data/sequences_false.csv"

preprocessed_data_file_path = "data/train_test_data.csv"

num_of_sequences = 5353

UNIQUE_CHARS = ['V', 'W', 'H', 'G', 'T', 'Q', 'A', 'C', 'S', 'K', 'Y', 'L', 'R', 'D', 'M', 'E', 'I', 'F', 'N', 'P', '-', 'X']
# These dictionaries will be updated when running main()
PROTEIN_ENCODING = dict()
BINARY_ENCODING = dict()
