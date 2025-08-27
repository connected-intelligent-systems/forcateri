from forcateri.model.dartsmodels.dartstcnmodel import DartsTCNModel
from forcateri.baltbestapi.baltbestaggregatedapidata import BaltBestAggregatedAPIData
import pandas as pd
from forcateri.data.dataprovider import DataProvider, SeriesRole
# from darts.models import TCNModel
# from darts.utils.likelihood_models import QuantileRegression
from forcateri.data.timeseries import TimeSeries
from forcateri.reporting.dimwiseaggregatedmetric import DimwiseAggregatedMetric
from forcateri.reporting.dimwiseaggregatedquantileloss import DimwiseAggregatedQuantileLoss
from forcateri.reporting.resultreporter import ResultReporter
from forcateri.controls.pipeline import Pipeline


OFFSET, TIME_STEP = TimeSeries.ROW_INDEX_NAMES
FEATURE, REPRESENTATION = TimeSeries.COL_INDEX_NAMES

def main():
    ds0 = BaltBestAggregatedAPIData()
    roles = {
        'q_hca': SeriesRole.TARGET, 
        'temperature_outdoor_avg':SeriesRole.KNOWN, 
        'temperature_1_max':SeriesRole.OBSERVED, 
        'temperature_2_max':SeriesRole.OBSERVED,
        'temperature_room_avg':SeriesRole.OBSERVED,}
    dp = DataProvider(data_sources=[ds0], roles=roles)

    mad0 = DartsTCNModel()
    #met0 = DimwiseAggregatedQuantileLoss(axes=[OFFSET])
    #met0 = DimwiseAggregatedMetric(axes=[OFFSET])
    met1 = DimwiseAggregatedMetric(axes=[TIME_STEP])
    test_set = dp.get_test_set()
    rep = ResultReporter(test_set,[mad0],[met1])
    #rep.report_all()

    pipe = Pipeline(dp,model_adapter=mad0,reporter=rep)
    results = pipe.run()
    return results

if __name__ == "__main__":
    main()
