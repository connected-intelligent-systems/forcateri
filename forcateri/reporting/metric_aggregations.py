import numpy as np


def column_wise_mae(ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    err = ground_truth - prediction
    sq_err = np.abs(err)
    mse = sq_err.mean(axis=0)
    return mse


def quantile_metric(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    quantiles = np.asarray(quantiles)
    diff = ground_truth - prediction
    loss = np.maximum(quantiles * diff, (quantiles - 1) * diff)
    return np.mean(loss, axis=0)
