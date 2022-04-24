import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_diff(y):
    return y * (1 - y)


def tanh(x):
    return np.tanh(x)


def tanh_diff(y):
    return 1 - y * y


def create_sequences(data, window_size):
    x, y = [], []
    for i in data.index:
        if i + window_size + 1 > data.index[-1]:
            break
        x.append(data.iloc[i:i+window_size])
        y.append(data.iloc[i+window_size+1])
    return np.array(x), np.array(y)


def split_data(x, y, split_ratio):
    total_len = x.shape[0]
    split = int(total_len*split_ratio)
    return x[:split], x[split:], y[:split], y[split:]