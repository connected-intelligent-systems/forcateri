from ..data.timeseries import TimeSeries
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class Metric:

    def align(self, ts_gt: TimeSeries, ts_pred: TimeSeries):

        horizon = ts_pred.offsets.max() // pd.Timedelta(1, ts_pred.freq)
        logger.debug(
            f"While computing metric the following horizon was calculated on pred_ts: {horizon}"
        )

        if horizon < 1:
            raise ValueError(
                f"Invalid model adapter output. "
                f"Horizon is expected to be 1 or greater but was {horizon}."
            )

        ts_gt_shifted = ts_gt.shift_to_repeat_to_multihorizon(horizon=horizon)
        logger.debug(f"\ngt_shifted:\n{ts_gt_shifted}")
        common_index = ts_gt_shifted.data.index.intersection(ts_pred.data.index)
        logger.debug(
            f"Common index determined to be\n{common_index.to_frame(index=False)}"
        )
        ts_gt_shifted.data = ts_gt_shifted.data.loc[common_index]
        old_pred_len = len(ts_pred)
        ts_pred.data = ts_pred.data.loc[common_index]
        dropped_gt_steps = len(ts_gt) - len(ts_gt_shifted)
        dropped_pred_steps = old_pred_len - len(ts_pred)
        if (dropped_gt_steps, dropped_pred_steps) != (0, 0):
            logger.warning(
                f"Alignment dropped {dropped_gt_steps} time steps ftom the ground truth "
                f"and {dropped_pred_steps} time steps from the prediction."
            )
        else:
            logger.debug("No time steps were dropped during alignment.")
        return ts_gt_shifted, ts_pred

    def __call__(self, gt: TimeSeries, pred: TimeSeries):
        raise NotImplementedError("Must be overridden in child classes.")
