import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.models import DeepFM
from conf import TARGET, DENSE_FEATURES, SPARSE_FEATURES, FILE_ROWS,\
    TRAIN_FILES, VALID_FILES, RANDOM_SEED, BATCH_SIZE, EPOCHS, EARLY_STOP


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, files, feature_names, minmax_scaler, ordinal_encoder,
                 shuffle=True):
        self.files = files
        self.feature_names = feature_names
        self.minmax_scaler = minmax_scaler
        self.ordinal_encoder = ordinal_encoder
        self.X = None
        self.y = None
        self.indexes = np.arange(len(self.files))
        self.shuffle = shuffle

    def __len__(self):
        return len(self.files) * FILE_ROWS // BATCH_SIZE

    def __getitem__(self, index):
        start = index * BATCH_SIZE % FILE_ROWS
        if start == 0:
            file = self.files[self.indexes[index * BATCH_SIZE // FILE_ROWS]]
            print(f'\nread file: {file}\n')
            df = pd.read_csv(
                file, delimiter='\t', header=None, nrows=FILE_ROWS,
                names=[TARGET] + DENSE_FEATURES + SPARSE_FEATURES)

            df[DENSE_FEATURES] = self.minmax_scaler.transform(
                df[DENSE_FEATURES])
            df[DENSE_FEATURES] = df[DENSE_FEATURES].fillna(0)
            df[SPARSE_FEATURES] = df[SPARSE_FEATURES].fillna('')
            df[SPARSE_FEATURES] = self.ordinal_encoder.transform(
                df[SPARSE_FEATURES]) + 1

            self.X = df[DENSE_FEATURES + SPARSE_FEATURES]
            self.y = df[[TARGET]]

        X = self.X.iloc[start:start + BATCH_SIZE, :]
        y = self.y.iloc[start:start + BATCH_SIZE, :]
        return [X[name].values for name in self.feature_names], y.values

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def train():
    minmax_scaler = joblib.load('./output/minmax_scaler')
    ordinal_encoder = joblib.load('./output/ordinal_encoder')

    feature_columns = [SparseFeat(
        x, vocabulary_size=ordinal_encoder.categories_[i].shape[0] + 1,
        embedding_dim=10) for i, x in enumerate(SPARSE_FEATURES)
    ] + [DenseFeat(x, dimension=1) for x in DENSE_FEATURES]

    if RANDOM_SEED:
        seed = RANDOM_SEED
        np.random.seed(seed)
        tf.random.set_seed(seed)
    else:
        seed = np.random.randint(0, 1e10)

    model = DeepFM(feature_columns, feature_columns, task='binary', seed=seed)
    model.compile('adam', 'binary_crossentropy', metrics=['AUC'])

    feature_names = get_feature_names(feature_columns)
    train_data = DataGenerator(
        TRAIN_FILES, feature_names, minmax_scaler, ordinal_encoder)
    valid_data = DataGenerator(
        VALID_FILES, feature_names, minmax_scaler, ordinal_encoder, False)

    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=EARLY_STOP, restore_best_weights=True)
    model.fit(train_data, batch_size=BATCH_SIZE, epochs=EPOCHS,
              validation_data=valid_data, callbacks=[early_stop])
    model.save('./output/model.h5')


if __name__ == '__main__':
    train()
