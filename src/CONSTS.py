
raw_data_file_path = "data/NoGapsMSA_SIS1-on-top.fasta"

seq_true_file_path = "data/sequences_true.csv"
seq_false_file_path = "data/sequences_false.csv"

train_test_sample_file_path = "data/train_test_data.csv"
pereprocessed_data_file_path = "data/encoded.csv"

NUM_OF_SEQUENCES = 5353

UNIQUE_CHARS = ['V', 'W', 'H', 'G', 'T', 'Q', 'A', 'C', 'S', 'K', 'Y', 'L', 'R', 'D', 'M', 'E', 'I', 'F', 'N', 'P', '-', 'X']
# These dictionaries will be updated when running main()
PROTEIN_ENCODING = dict()
BINARY_ENCODING = dict()

DF_SEQ_COLUMNS = ["names", "seq_encoded", "label"]
