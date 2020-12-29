import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from absl.flags import FLAGS
from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names
from deepctr.layers import custom_objects
from deepctr.models import DeepFM


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, files, feature_names, minmax_scaler, ordinal_encoder,
                 shuffle=True):
        self.files = files
        self.feature_names = feature_names
        self.minmax_scaler = minmax_scaler
        self.ordinal_encoder = ordinal_encoder
        self.file = None
        self.X = None
        self.y = None
        self.shuffle = shuffle

    def __len__(self):
        return len(self.files) * FLAGS.file_rows // FLAGS.batch_size

    def __getitem__(self, index):
        file = self.files[index * FLAGS.batch_size // FLAGS.file_rows]
        if self.file != file:
            self.file = file
            df = pd.read_csv(
                file, delimiter='\t', header=None, nrows=FLAGS.file_rows,
                names=[FLAGS.target] + FLAGS.dense_feat + FLAGS.sparse_feat)

            df[FLAGS.dense_feat] = self.minmax_scaler.transform(
                df[FLAGS.dense_feat].fillna(0))
            df[FLAGS.sparse_feat] = self.ordinal_encoder.transform(
                df[FLAGS.sparse_feat].fillna('')) + 1

            self.X = df[FLAGS.dense_feat + FLAGS.sparse_feat]
            self.y = df[[FLAGS.target]]

            self.indexes = np.arange(FLAGS.file_rows // FLAGS.batch_size)
            if self.shuffle:
                np.random.shuffle(self.indexes)

        start = self.indexes[
            index % (FLAGS.file_rows // FLAGS.batch_size)] * FLAGS.batch_size
        X = self.X.iloc[start:start + FLAGS.batch_size, :]
        y = self.y.iloc[start:start + FLAGS.batch_size, :]
        return [X[name].values for name in self.feature_names], y.values

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)


def main():
    minmax_scaler = joblib.load('./output/minmax_scaler')
    ordinal_encoder = joblib.load('./output/ordinal_encoder')

    feature_columns = [SparseFeat(
        x, vocabulary_size=ordinal_encoder.categories_[i].shape[0] + 1,
        embedding_dim=10) for i, x in enumerate(FLAGS.sparse_feat)
    ] + [DenseFeat(x, dimension=1) for x in FLAGS.dense_feat]

    if FLAGS.random_seed:
        seed = FLAGS.random_seed
        np.random.seed(seed)
        tf.random.set_seed(seed)
    else:
        seed = np.random.randint(0, 1e10)

    if FLAGS.continue_train:
        model = tf.keras.models.load_model('./output/model', custom_objects)
    else:
        model = DeepFM(
            feature_columns, feature_columns, task='binary', seed=seed)
        model.compile('adam', 'binary_crossentropy', metrics=['AUC'])

    feature_names = get_feature_names(feature_columns)
    train_data = DataGenerator(
        FLAGS.train_files, feature_names, minmax_scaler, ordinal_encoder)
    valid_data = DataGenerator(
        FLAGS.valid_files, feature_names, minmax_scaler, ordinal_encoder,
        False)

    early_stop = tf.keras.callbacks.EarlyStopping(
        patience=FLAGS.early_stop, restore_best_weights=True)
    model.fit(train_data, validation_data=valid_data, callbacks=[early_stop],
              shuffle=False, batch_size=FLAGS.batch_size, epochs=FLAGS.epochs)
    model.save('./output/model')
