import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mase(y_true, y_pred, y_train, m=24):
    naive_errors = np.abs(y_train[m:] - y_train[:-m])
    scale = np.mean(naive_errors)
    return np.mean(np.abs(y_true - y_pred)) / (scale + 1e-8)
