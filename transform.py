import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from conf import TARGET, DENSE_FEATURES, SPARSE_FEATURES, FILE_ROWS,\
    TRAIN_FILES


def transform():
    minmaxs = [[float('inf'), float('-inf')] for _ in DENSE_FEATURES]
    vocabs = [set(['']) for _ in SPARSE_FEATURES]

    for file in TRAIN_FILES:
        print(f'process file: {file}')
        data = pd.read_csv(
            file, delimiter='\t', header=None, nrows=FILE_ROWS,
            names=[TARGET] + DENSE_FEATURES + SPARSE_FEATURES)

        data[DENSE_FEATURES] = data[DENSE_FEATURES].fillna(0)
        data[SPARSE_FEATURES] = data[SPARSE_FEATURES].fillna('')

        for col, minmax in zip(DENSE_FEATURES, minmaxs):
            minmax[0] = min(minmax[0], data[col].min())
            minmax[1] = max(minmax[1], data[col].max())

        for col, vocab in zip(SPARSE_FEATURES, vocabs):
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


if __name__ == '__main__':
    transform()
