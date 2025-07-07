from forcateri.model.dartsmodels.dartstcnmodel import DartsTCNModel
from forcateri.baltbestapi.baltbestaggregatedapidata import BaltBestAggregatedAPIData
import pandas as pd
from forcateri.data.dataprovider import DataProvider, SeriesRole
from darts.models import TCNModel
from darts.utils.likelihood_models import QuantileRegression
from forcateri.data.timeseries import TimeSeries

roles = {
    'q_hca': SeriesRole.TARGET, 
    'temperature_outdoor_avg':SeriesRole.KNOWN, 
    'temperature_1_max':SeriesRole.OBSERVED, 
    'temperature_2_max':SeriesRole.OBSERVED,
    'temperature_room_avg':SeriesRole.OBSERVED,}

baltbest = BaltBestAggregatedAPIData()
dataprovider = DataProvider(data_sources=[baltbest], roles=roles)
train = dataprovider.get_train_set()
val = dataprovider.get_val_set()
test = dataprovider.get_test_set()
quantiles = [0.1, 0.5, 0.9]


dartstcn = DartsTCNModel.load( "/home/user/DFKI/forcateri/_data/dartstcn.pt")
#dartstcn.fit(train_data = train, val_data=val)

num_forecast_steps = 2
#num_monte_carlo_samples = 50 # A sufficient number of samples to approximate the quantiles
dartstcn.quantiles = quantiles
#Here prediction is darts timeseries
prediction = dartstcn.predict(
    data=val, # Pass the series used for prediction context
    predict_likelihood_parameters = True,
    n=num_forecast_steps,
)
prediction_ts = dartstcn.to_time_series(prediction)
print(prediction_ts[0].data)
#dartstcn.save("/home/user/DFKI/forcateri/_data/dartstcn.pt")