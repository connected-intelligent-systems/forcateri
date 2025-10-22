import numpy as np


def column_wise_mae(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    err = gt - pred
    sq_err = np.abs(err)
    mse = sq_err.mean(axis=0)
    return mse


def quantile_metric(
    gt: np.ndarray,
    pred: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    quantiles = np.asarray(quantiles)
    diff = gt - pred
    loss = np.maximum(quantiles * diff, (quantiles - 1) * diff)
    return np.mean(loss, axis=0)
