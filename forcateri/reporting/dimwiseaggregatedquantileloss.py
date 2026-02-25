from typing import List, Optional
import logging
from .dimwiseaggregatedmetric import DimwiseAggregatedMetric
from .metric_aggregations import quantile_metric
from ..data.timeseries import TimeSeries
from .metric import Metric
import numpy as np

logger = logging.getLogger(__name__)


class DimwiseAggregatedQuantileLoss(DimwiseAggregatedMetric):
    def __init__(
        self,
        axes: List[str],
        name: Optional[str] = None,
    ):
        self.axes = axes
        super().__init__(name=name or str(self), axes=axes)

    def __call__(self, ground_truth: TimeSeries, prediction: TimeSeries):

        if prediction.quantiles is None:
            logger.error(
                "Predicted TimeSeries must have quantiles defined for DimwiseAggregatedQuantileLoss."
            )
            raise ValueError(
                "Predicted TimeSeries must have quantiles defined for DimwiseAggregatedQuantileLoss."
            )
        def quantile_loss(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
            return quantile_metric(gt, pred, prediction.quantiles)

        self.reduction = quantile_loss
        return super().__call__(ground_truth, prediction)


    def __str__(self):
        """
        Return string representation.

        See `DimwiseAggregatedMetric.__str__` for full details.
        """
        return super().__str__()
