TARGET = 'label'
DENSE_FEATURES = [f'I{i}' for i in range(1, 14)]
SPARSE_FEATURES = [f'C{i}' for i in range(1, 27)]

FILE_ROWS = 500000
TRAIN_FILES = [f'./data/train_{i:0>2d}' for i in range(2)]
VALID_FILES = [f'./data/train_{i:0>2d}' for i in range(80, 81)]

RANDOM_SEED = 2020
BATCH_SIZE = 50
EPOCHS = 10
EARLY_STOP = 2
