from ..data.timeseries import TimeSeries


class Metric:

    def __call__(self, gt: TimeSeries, pred: TimeSeries):
        raise NotImplementedError("Must be overridden in child classes.")
