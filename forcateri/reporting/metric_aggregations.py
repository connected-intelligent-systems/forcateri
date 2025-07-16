import numpy as np


def column_wise_mae(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    err = gt - pred
    sq_err = np.abs(err)
    mse = sq_err.mean(axis=0)
    return mse
