import joblib
import pandas as pd
from category_encoders import OrdinalEncoder
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from conf import TARGET, DENSE_FEATURES, SPARSE_FEATURES, TRAIN_FILES, VALID_FILES


def transform():
    minmaxs = [[float('inf'), float('-inf')] for _ in DENSE_FEATURES]
    vocabs = [set() for _ in SPARSE_FEATURES]

    for file in TRAIN_FILES:
        print(f'process file: {file}')
        data = pd.read_csv(
            file, delimiter='\t', header=None,
            names=[TARGET] + DENSE_FEATURES + SPARSE_FEATURES)

        data[DENSE_FEATURES] = data[DENSE_FEATURES].fillna(0)
        data[SPARSE_FEATURES] = data[SPARSE_FEATURES].fillna('')

        for col, minmax in zip(DENSE_FEATURES, minmaxs):
            this_min = data[col].min()
            if minmax[0] > this_min:
                minmax[0] = this_min

            this_max = data[col].max()
            if minmax[1] < this_max:
                minmax[1] = this_max

        for col, vocab in zip(SPARSE_FEATURES, vocabs):
            for x in data[col].unique():
                vocab.add(x)

    minmax_scaler = MinMaxScaler()
    minmax_scaler.fit(pd.DataFrame(minmaxs).transpose())
    joblib.dump(minmax_scaler, './minmax_scaler')

    mapping = [{
        'col': col,
        'mapping': pd.Series(range(len(vocab)), index=vocab)
    } for col, vocab in zip(SPARSE_FEATURES, vocabs)]
    ordinal_encoder = OrdinalEncoder(
        mapping=mapping, cols=SPARSE_FEATURES, handle_unknown='return_nan',
        handle_missing='return_nan')
    ordinal_encoder.fit(pd.DataFrame([], columns=SPARSE_FEATURES))
    joblib.dump(ordinal_encoder, './ordinal_encoder')


if __name__ == '__main__':
    transform()
