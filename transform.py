import joblib
import numpy as np
import pandas as pd
from absl.flags import FLAGS
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


def main():
    minmaxs = [[float('inf'), float('-inf')] for _ in FLAGS.dense_feat]
    vocabs = [set(['']) for _ in FLAGS.sparse_feat]

    for file in FLAGS.train_files:
        print(f'process file: {file}')
        data = pd.read_csv(
            file, delimiter='\t', header=None, nrows=FLAGS.file_rows,
            names=[FLAGS.target] + FLAGS.dense_feat + FLAGS.sparse_feat)

        data[FLAGS.dense_feat] = data[FLAGS.dense_feat].fillna(0)
        data[FLAGS.sparse_feat] = data[FLAGS.sparse_feat].fillna('')

        for col, minmax in zip(FLAGS.dense_feat, minmaxs):
            minmax[0] = min(minmax[0], data[col].min())
            minmax[1] = max(minmax[1], data[col].max())

        for col, vocab in zip(FLAGS.sparse_feat, vocabs):
            vocab.update(data[col].unique())

    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(pd.DataFrame(minmaxs).transpose())
    joblib.dump(minmax_scaler, './output/minmax_scaler')

    ordinal_encoder = OrdinalEncoder(
        dtype=np.int32, handle_unknown='use_encoded_value', unknown_value=-1)
    vocabs = pd.concat([pd.Series(list(vocab)) for vocab in vocabs], axis=1)
    vocabs = vocabs.fillna('')
    ordinal_encoder.fit(vocabs)
    joblib.dump(ordinal_encoder, './output/ordinal_encoder')
