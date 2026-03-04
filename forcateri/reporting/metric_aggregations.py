import numpy as np

def column_wise_mape(
    ground_truth: np.ndarray, 
    prediction: np.ndarray, 
    epsilon: float = 1e-8
) -> np.ndarray:
    
    #Handling the cases where ground truth is 0.
    denom = np.where(ground_truth == 0, epsilon, ground_truth)
    err = np.abs((ground_truth - prediction) / denom)
    
    mape = np.mean(err, axis=0) * 100
    return mape

def column_wise_wmape(
    ground_truth: np.ndarray, 
    prediction: np.ndarray, 
    epsilon: float = 1e-8
) -> np.ndarray:
    
    numerator = np.sum(np.abs(ground_truth - prediction), axis=0)
    denominator = np.sum(np.abs(ground_truth), axis=0)
    
    #the case where ground truth is zero.
    denominator = np.where(denominator == 0, epsilon, denominator)
    
    return (numerator / denominator) * 100
    

def column_wise_mae(ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    err = ground_truth - prediction
    abs_err = np.abs(err)
    mae = abs_err.mean(axis=0)
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
