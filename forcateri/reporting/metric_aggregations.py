import logging

import numpy as np


logger = logging.getLogger(__name__)


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
    logger.debug(f"Quantiles in quantile metric computation:\n{quantiles}")
    logger.debug(f"Ground truth in quantile metric computation:\n{gt}")
    logger.debug(f"Prediction in quantile metric computation:\n{pred}")
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    quantiles = np.asarray(quantiles)
    diff = gt - pred
    logger.debug(f"Error in quantile metric computation:\n{diff}")
    loss = np.maximum(quantiles * diff, (quantiles - 1) * diff)
    logger.debug(f"Loss in quantile metric computation:\n{loss}")
    return np.mean(loss, axis=0)
