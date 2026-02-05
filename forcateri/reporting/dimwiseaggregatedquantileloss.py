from typing import List
import logging
from .dimwiseaggregatedmetric import DimwiseAggregatedMetric
from .metric_aggregations import quantile_metric
from ..data.timeseries import TimeSeries
from .metric import Metric

logger = logging.getLogger(__name__)


class DimwiseAggregatedQuantileLoss(DimwiseAggregatedMetric):
    def __init__(
        self,
        axes: List[str],
    ):
        super().__init__(axes, None)

    def __call__(self, ts_gt: TimeSeries, ts_pred: TimeSeries):
        ts_gt, ts_pred = Metric.align(ts_gt, ts_pred)  # for alignment 
        if ts_pred.quantiles is None:
            logger.error(
                "Predicted TimeSeries must have quantiles defined for DimwiseAggregatedQuantileLoss."
            )
            raise ValueError(
                "Predicted TimeSeries must have quantiles defined for DimwiseAggregatedQuantileLoss."
            )
        self.reduction = lambda gt, pred: quantile_metric(gt, pred, ts_pred.quantiles)
        return super().__call__(ts_gt, ts_pred)

    def __str__(self):
        axes_str = "_".join(map(str, self.axes))
        return f"{self.__class__.__name__}_on_{axes_str}_using_quantile_metric"
