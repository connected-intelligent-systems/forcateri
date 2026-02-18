from typing import List, Optional
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
        name: Optional[str],
    ):
        super().__init__(name=name or str(self),axes=axes)

    def __call__(self, ground_truth: TimeSeries, prediction: TimeSeries):

        if prediction.quantiles is None:
            logger.error(
                "Predicted TimeSeries must have quantiles defined for DimwiseAggregatedQuantileLoss."
            )
            raise ValueError(
                "Predicted TimeSeries must have quantiles defined for DimwiseAggregatedQuantileLoss."
            )
        self.reduction = lambda gt, pred: quantile_metric(gt, pred, prediction.quantiles)
        return super().__call__(ground_truth, prediction)

    def __str__(self):
        axes_str = "_".join(map(str, self.axes))
        return f"{self.__class__.__name__}_on_{axes_str}_using_quantile_metric"
