from typing import Callable, List, Union, Optional
import numpy as np
import pandas as pd
import logging

from .metric import Metric
from .metric_aggregations import column_wise_mae
from ..data.timeseries import TimeSeries

logger = logging.getLogger(__name__)


class DimwiseAggregatedMetric(Metric):
    """Applies a reduction function across a given set of axes.

    Produces a data frame with one reduced value,
    i.e., the return value of the reduction function,
    for every index of every unreduced axis.
    If reduced across all axes, the result
    is the total reduced value insetad (usually a scalar).
    If there is more than one unreduced axis
    or the reduction function has a non-scalar output,
    the resulting data frame has a MultiIndex.

    For shorter and more informative string representations
    we use the following convention for axis abbreviation:
        T: time
        O: offset
        F: feature
    """

    OFFSET, TIME = TimeSeries.ROW_INDEX_NAMES
    FEATURE, REPRESENTATION = TimeSeries.COL_INDEX_NAMES

    def __init__(
        self,
        axes: List[str],
        reduction: Callable[
            [np.ndarray, np.ndarray], Union[np.ndarray, float]
        ] = column_wise_mae,
        name: Optional[str] = None,
    ):
        self.axes = axes
        self.reduction = reduction
        super().__init__(name or str(self))

    @staticmethod
    def get_level_values(df, axis):
        if axis in df.index.names:
            return df.index.get_level_values(axis)
        elif axis in df.columns.names:
            return df.columns.get_level_values(axis)
        else:
            raise ValueError("Axis not found neither in row nor in column index.")

    def __call__(self, ground_truth: TimeSeries, prediction: TimeSeries):

        ground_truth, prediction = Metric.align(ground_truth, prediction)
        flat_pred = prediction.data.copy().stack(level=0, future_stack=True)
        flat_gt = ground_truth.data.copy().stack(level=0, future_stack=True)
        logger.info(f"Reducing axes {self.axes}")
        group_by = sorted(
            list(
                {
                    DimwiseAggregatedMetric.OFFSET,
                    DimwiseAggregatedMetric.TIME,
                    DimwiseAggregatedMetric.FEATURE,
                    DimwiseAggregatedMetric.REPRESENTATION,
                }
                - set(
                    [*self.axes, DimwiseAggregatedMetric.REPRESENTATION]
                )  # representation gets special treatment because of possible dimension mismatch
            )
        )

        logger.debug(f"Grouping by {group_by}")

        if len(group_by) == 0:
            logger.info("No axes left for grouping. Reducing entire data frames.")

            reduced = self.reduction(flat_gt.values, flat_pred.values)
            return pd.DataFrame(
                data=reduced.reshape(1, 2),
                columns=DimwiseAggregatedMetric.get_level_values(
                    flat_pred, DimwiseAggregatedMetric.REPRESENTATION
                ),
            )

        else:
            logger.info(f"=> grouping_by {group_by}")

            reduced_index = pd.MultiIndex.from_product(
                [
                    DimwiseAggregatedMetric.get_level_values(
                        ground_truth.data, axis
                    ).unique()
                    for axis in group_by
                ]
            )
            if len(reduced_index.levels) == 1:
                reduced_index = reduced_index.levels[0]
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
                logger.debug(f"gt:{gt_label}, pred_label: {pred_label}")
                assert (
                    gt_label == pred_label
                )  # due to the identical structure before grouping and the same group_by
                reduced = self.reduction(gt.values, pred.values)
                reduced_df.loc[pred_label] = reduced
                logger.debug(f"\ngt:\n{gt}\npred:\n{pred}")
                logger.debug(f"Reduced:{reduced}")
            return reduced_df

    def __str__(self):
        """
        Return a compact string identifier describing the aggregation.

        Format:
            DimwAgg_on_<axis>_using_<reduction>

        where:
            axes_abr : first char of each axis
            reduction : name of the reduction function
        """
        axes_abr = "".join(str(a)[0] for a in self.axes).upper()
        return f"DimwAgg_{axes_abr}_{self.reduction.__name__}"
