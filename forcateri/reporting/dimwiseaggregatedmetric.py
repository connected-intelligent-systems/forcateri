from typing import Callable, List, OrderedDict, Union
import numpy as np
import pandas as pd

from .metric import Metric
from ..data.timeseries import TimeSeries


class DimwiseAggregatedMetric(Metric):
    OFFSET, TIME_STEP = TimeSeries.ROW_INDEX_NAMES
    FEATURE, REPRESENTATION = TimeSeries.COL_INDEX_NAMES

    def __init__(
        self,
        axes: List[str],
        reduction: Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]],
    ):
        self.axes = axes
        self.reduction = reduction

    @staticmethod
    def get_level_values(df, axis):
        if axis in df.index.names:
            return df.index.get_level_values(axis)
        elif axis in df.columns.names:
            return df.columns.get_level_values(axis)
        else:
            raise ValueError("Axis not found neither in row nor in column index.")

    def __call__(self, ts_gt: TimeSeries, ts_pred: TimeSeries):
        flat_pred = ts_pred.data.copy().stack(level=0, future_stack=True)
        flat_gt = ts_gt.data.copy().stack(level=0, future_stack=True)
        print(f"Reducing axes {self.axes}")
        group_by = sorted(
            list(
                {
                    DimwiseAggregatedMetric.OFFSET,
                    DimwiseAggregatedMetric.TIME_STEP,
                    DimwiseAggregatedMetric.FEATURE,
                    DimwiseAggregatedMetric.REPRESENTATION,
                }
                - set(
                    [*self.axes, DimwiseAggregatedMetric.REPRESENTATION]
                )  # representation gets special treatment because of possible dimension mismatch
            )
        )

        if len(group_by) == 0:
            print("No axes left for grouping. Reducing entire data frames.")
            reduced = self.reduction(flat_gt.values, flat_pred.values)
            return pd.DataFrame(
                data=reduced.reshape(1, 2),
                columns=DimwiseAggregatedMetric.get_level_values(
                    flat_pred, DimwiseAggregatedMetric.REPRESENTATION
                ),
            )

        else:
            print(f"=> grouping_by {group_by}")

            reduced_index = pd.MultiIndex.from_product(
                [
                    DimwiseAggregatedMetric.get_level_values(ts_gt.data, axis).unique()
                    for axis in group_by
                ]
            )
            reduced_df = pd.DataFrame(
                index=reduced_index,
                columns=DimwiseAggregatedMetric.get_level_values(
                    flat_pred, DimwiseAggregatedMetric.REPRESENTATION
                ),  # quantile loss would have only one column
            )

            for (
                (gt_label, gt),
                (pred_label, pred),
            ) in zip(flat_gt.groupby(group_by), flat_pred.groupby(group_by)):
                assert (
                    gt_label == pred_label
                )  # due to the identical structure before grouping and the same group_by
                reduced = self.reduction(gt.values, pred.values)
                reduced_df.loc[pred_label] = reduced

            return reduced_df
