TARGET = 'label'
DENSE_FEATURES = [f'I{i}' for i in range(1, 14)]
SPARSE_FEATURES = [f'C{i}' for i in range(1, 27)]

FILE_ROWS = 500  # 500000
TRAIN_FILES = [f'./data/train_{i:0>2d}' for i in range(20)]
VALID_FILES = [f'./data/train_{i:0>2d}' for i in range(20, 22)]

BATCH_SIZE = 50
EPOCHS = 10
EARLY_STOP = 2
