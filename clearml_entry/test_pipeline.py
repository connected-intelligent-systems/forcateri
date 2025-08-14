from forcateri.model.dartsmodels.dartstcnmodel import DartsTCNModel
from forcateri.baltbestapi.baltbestaggregatedapidata import BaltBestAggregatedAPIData
import pandas as pd
from forcateri.data.dataprovider import DataProvider, SeriesRole
# from darts.models import TCNModel
# from darts.utils.likelihood_models import QuantileRegression
from forcateri.data.timeseries import TimeSeries
from forcateri.reporting.dimwise_aggregated_quantile_loss import DimwiseAggregatedQuantileLoss
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

    met0 = DimwiseAggregatedQuantileLoss(axes=[OFFSET])

    rep = ResultReporter(dp.get_test_set(),[mad0],[met0])
    #rep.report_all()

    pipe = Pipeline(dp,mad=mad0,rep=rep)
    pipe.run()
