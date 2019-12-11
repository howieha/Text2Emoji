import pandas as pd
import numpy as np
import pickle
from scipy.sparse import *
from sklearn.model_selection import train_test_split


SEED = 5525

def update_index(df):
    index_set = set()
    for i in df.tolist():
        index_set.update(set(i))
    indices = list(index_set)
    indices.sort()
    return indices

def split_dataset(dataset):
    with open(dataset, 'rb') as handle:
        df = pickle.load(handle)
    #df = df.iloc[:10000,]
    df = df[df['TEXT_ID'].map(len)>=2]
    X = df.iloc[:, 0]
    Y = df.iloc[:, 1]
    indices = update_index(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=SEED)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=SEED)
    return X_train, X_test, X_val, Y_train, Y_test, Y_val, indices

def transfer_to_csr(raw_df, num_col, indices):
    shape_csr = (len(raw_df), num_col)
    row = []
    col = []
    row_idx = 0
    for emoji_ids in raw_df.tolist():
        tempEmoji = set(emoji_ids)
        row += [row_idx] * len(tempEmoji)
        idx = [indices.index(i) for i in emoji_ids]
        col += idx
        row_idx += 1
    data = [1]*len(row)
    return csr_matrix((data, (row, col)), shape=shape_csr)

def make_same_length(data, sequence_length):
    features = np.zeros((len(data), sequence_length), dtype=int)
    for i, text in enumerate(data):
        text_len = len(text)
        if (text_len <= sequence_length):
            zeros = list(np.zeros(sequence_length - text_len))
            new = text + zeros
        else:
            new = text[:sequence_length]
        features[i, :] = np.array(new)
    return features

def load_data_nn(dataset, num_class):
    X_train, X_test, X_val, Y_train, Y_test, Y_val, indices = split_dataset(dataset)
    X_train = make_same_length(X_train, 128)
    X_test = make_same_length(X_test, 128)
    X_val = make_same_length(X_val, 128)
    # Y_train = transfer_to_csr(Y_train, num_class, indices).todense()
    # Y_test = transfer_to_csr(Y_test, num_class, indices).todense()
    # Y_val = transfer_to_csr(Y_val, num_class, indices).todense()
    Y_train = transfer_to_csr(Y_train, num_class, indices)
    Y_test = transfer_to_csr(Y_test, num_class, indices)
    Y_val = transfer_to_csr(Y_val, num_class, indices)
    num_pos = np.sum(Y_train, axis=0) + np.sum(Y_test, axis=0) + np.sum(Y_val, axis=0)
    # weight_pos = ((len(Y_train) + len(Y_test) + len(Y_val)) - num_pos) / num_pos
    weight_pos = ((Y_train.shape[0] + Y_test.shape[0] + Y_val.shape[0]) - num_pos) / num_pos

    return X_train, X_test, X_val, Y_train, Y_test, Y_val, weight_pos

