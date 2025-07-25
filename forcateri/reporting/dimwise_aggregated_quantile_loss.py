from forcateri.reporting.dimwiseaggregatedmetric import DimwiseAggregatedMetric
from forcateri.reporting.metric_aggregations import quantile_metric


class DimwiseAggregatedQuantileLoss(DimwiseAggregatedMetric):
    def __init__(
        self,
        axes: List[str],
    ):
        super().__init__(axes, None)

    def __call__(self, ts_gt: TimeSeries, ts_pred: TimeSeries):
        self.reduction = lambda gt, pred: quantile_metric(gt, pred, ts_pred.quantiles)
        return super().__call__(ts_gt, ts_pred)
