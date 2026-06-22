import logging
from typing import Any, Tuple, Optional

import pandas as pd

from ..data.timeseries import TimeSeries

logger = logging.getLogger(__name__)


class Metric:

    def __init__(self, name: Optional[str], suppress_warnings:bool=False):
        
        self._name = name or self.__class__.__name__
        self._suppress_warnings = suppress_warnings

    @property
    def name(self):
        return self._name
    
    
    @staticmethod
    def align(ground_truth: TimeSeries, prediction: TimeSeries, suppress_warnings: bool = False) -> Tuple[TimeSeries, TimeSeries]:
        """Brings ground truth and prediction into a similar format.

        More precisely:
            1. Expands ground truth by creating a copy of its data for every offset of the prediction
            2. Shifts the time index and the offset to align the underlying data with the prediction
            3. Cuts away parts of the prediction for which there is no ground truth
            4. Cuts away parts of the ground truth for which there is no prediction

        Parameters
        ----------
        ground_truth : TimeSeries
            The ground truth time series
        prediction : TimeSeries
            The prediction time series
        suppress_warnings : bool, optional
            Whether to suppress alignment warnings, by default False

        Raises
        ------
        ValueError
            If the prediction offset is < 1
        """
        prediction = prediction.copy()
        horizon = prediction.offset.max() // pd.Timedelta(1, prediction.freq)
        logger.debug(
            f"While computing metric the following horizon was calculated on pred_ts: {horizon}"
        )

        if horizon < 1:
            raise ValueError(
                f"Invalid model adapter output. "
                f"Horizon is expected to be 1 or greater but was {horizon}."
            )

        ts_gt_shifted = ground_truth.shift_repeat_to_multihorizon(horizon=horizon,in_place=False)
        logger.debug(f"\ngt_shifted:\n{ts_gt_shifted}")
        common_index = ts_gt_shifted.data.index.intersection(prediction.data.index)
        logger.debug(
            f"Common index determined to be\n{common_index.to_frame(index=False)}"
        )
        ts_gt_shifted.data = ts_gt_shifted.data.loc[common_index]
        old_pred_len = len(prediction)
        old_gt_len = len(ts_gt_shifted)
        prediction.data = prediction.data.loc[common_index]
        dropped_gt_steps = old_gt_len - len(ts_gt_shifted)
        dropped_pred_steps = old_pred_len - len(prediction)
        if (dropped_gt_steps, dropped_pred_steps) != (0, 0) and not suppress_warnings:
            logger.warning(
                f"Alignment dropped {dropped_gt_steps} time steps from the ground truth "
                f"and {dropped_pred_steps} time steps from the prediction."
            )
        else:
            logger.debug("No time steps were dropped during alignment.")
        gt_features = ts_gt_shifted.data.columns.get_level_values('feature').unique()
        pred_features = prediction.data.columns.get_level_values('feature').unique()
        if len(gt_features) == len(pred_features) and not gt_features.equals(pred_features):
            logger.error(f"cannot align features with different names")
        return ts_gt_shifted, prediction

    def __call__(self, ground_truth: TimeSeries, prediction: TimeSeries) -> Any:
        """The intended way of trigering a metric computation.

        Makes the class callable.
        The actual computation logic is taken from the child classes.
        Consider adding `self.align` to the implementation
        to make metric instances more generalizable.
        """
        raise NotImplementedError("Must be overridden in child classes.")
