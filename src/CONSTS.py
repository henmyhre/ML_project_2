
DATA_PATH = "./data/"

RAW_DATA_FILE_PATH = DATA_PATH + "NoGapsMSA_SIS1-on-top.fasta"

SEQ_TRUE_FILE_PATH = DATA_PATH + "sequences_true.csv"
SEQ_FALSE_FILE_PATH = DATA_PATH + "sequences_false.csv"

TRAIN_TEST_DATA_PATH_BASE = DATA_PATH + "train_test_data_"
TRAIN_TEST_DATA_PATH_1_1 = TRAIN_TEST_DATA_PATH_BASE + "1_1.csv"
TRAIN_TEST_DATA_PATH_1_4 = TRAIN_TEST_DATA_PATH_BASE + "1_4.csv"
FILES = [TRAIN_TEST_DATA_PATH_1_1, TRAIN_TEST_DATA_PATH_1_4]


NUM_OF_SEQUENCES = 5353
UNIQUE_CHARS = ['V', 'W', 'H', 'G', 'T', 'Q', 'A', 'C', 'S', 'K', 'Y', 'L', 'R', 'D', 'M', 'E', 'I', 'F', 'N', 'P', '-', 'X']

# These dictionaries will be updated when running main()
PROTEIN_ENCODING = dict()
BINARY_ENCODING = dict()

DF_SEQ_COLUMNS = ["names", "seq_encoded", "label"]

NEURAL_NET_1 = "One layer neural net"
NEURAL_NET_2 = "Two layer neural net"
LOGISTIC_REGRESSION = "Logistic regression"
MODEL_TYPES = [LOGISTIC_REGRESSION, NEURAL_NET_1, NEURAL_NET_2]

LEARNING_RATES = [1e-1, 1e-3, 1e-4]

# Operations for comparing start/end sequences
ADD = "add"
MULTIPLY = "multiply"
CONCATENATE = "concatenate"
OPERATIONS = [ADD, MULTIPLY, CONCATENATE]

