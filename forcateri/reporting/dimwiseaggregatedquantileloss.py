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
        self.reduction = lambda gt, pred: quantile_metric(
            gt, pred, prediction.quantiles
        )
        return super().__call__(ground_truth, prediction)


    def __str__(self):
        """
        Return a compact string identifier describing the aggregation.

        Format:
            DimwAgg_on_<axis>_using_<reduction>

        where:
            axes_abr : first char of each axis
            reduction : name of the reduction function

        Axis abbreviations:
            T = time
            O = offset
            F = feature
        """
        axes_abr = "".join(str(a)[0] for a in self.axes)
        return f"DimwAgg_{axes_abr}_quantile_metric_{self.reduction.__name__}"
