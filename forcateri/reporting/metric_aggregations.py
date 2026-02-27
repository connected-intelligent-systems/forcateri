import numpy as np

def column_wise_mape(ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    err = np.abs((ground_truth - prediction) / ground_truth)

    # avoid division by zero
    err = np.where(ground_truth == 0, np.nan, err)

    mape = np.mean(err, axis=0) * 100
    return mape


def column_wise_mae(ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    err = ground_truth - prediction
    sq_err = np.abs(err)
    mae = sq_err.mean(axis=0)
    return mae


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
