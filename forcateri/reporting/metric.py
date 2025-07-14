import numpy as np
from typing import Callable

class Metric:
    
    def __init__(self, func: str):
        self.func = self._get_func(func)  # store the specific metric function

    
    def __call__(self,gt:np.ndarray,pred:np.ndarray):
        return self.func(gt,pred)
    
    @staticmethod
    def _get_func(name:str) -> Callable:
        match name:
            case 'mae':
                return Metric.column_wise_mae
            case _:
                raise ValueError(f"Unknown metric name: {name}")

    @staticmethod
    def column_wise_mae(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
        err = gt - pred
        sq_err = np.abs(err)
        mse = sq_err.mean(axis=0)
        return mse