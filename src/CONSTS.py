
raw_data_file_path = "data/NoGapsMSA_SIS1-on-top.fasta"

seq_true_file_path = "data/sequences_true.csv"
seq_false_file_path = "data/sequences_false.csv"

# Model input filenames
TRAIN_TEST_DATA_PATH_1_1 = "data/train_test_data_1_1.csv"
TRAIN_TEST_DATA_PATH_1_4 = "data/train_test_data_1_4.csv"
FILES = [TRAIN_TEST_DATA_PATH_1_1, TRAIN_TEST_DATA_PATH_1_4]


NUM_OF_SEQUENCES = 5353

UNIQUE_CHARS = ['V', 'W', 'H', 'G', 'T', 'Q', 'A', 'C', 'S', 'K', 'Y', 'L', 'R', 'D', 'M', 'E', 'I', 'F', 'N', 'P', '-', 'X']
# These dictionaries will be updated when running main()
PROTEIN_ENCODING = dict()
BINARY_ENCODING = dict()

DF_SEQ_COLUMNS = ["names", "seq_encoded", "label"]



# Optimize learning technique
LR = [1, 1e-1, 1e-3, 1e-4, 1e-5]
NEURAL_NET_1 = "One layer neural net"
NEURAL_NET_2 = "Two layer neural net"
LOGISTIC_REGRESSION = "Logistic regression"
MODEL_TYPE = [LOGISTIC_REGRESSION, NEURAL_NET_1, NEURAL_NET_2]

ADD = "add"
MULTIPLY = "multiply"
CONCATENATE = "concatenate"
COMPARE = [ADD, MULTIPLY, CONCATENATE]

