from .dartsmodels.dartstcnmodel import DartsTCNModel
from ..baltbestapi.baltbestaggregatedapidata import BaltBestAggregatedAPIData
import pandas as pd 
from ..data.dataprovider import DataProvider, SeriesRole
#from darts.models import TCNModel 
from ..data.timeseries import TimeSeries
from darts.models import TCNModel
import pickle

baltbest = BaltBestAggregatedAPIData() 
roles = {
    'q_hca': SeriesRole.TARGET, 
    'temperature_outdoor_avg':SeriesRole.KNOWN, 
    'temperature_1_max':SeriesRole.OBSERVED, 
    'temperature_2_max':SeriesRole.OBSERVED,
    'temperature_room_avg':SeriesRole.OBSERVED,}
dataprovider = DataProvider(data_sources=[baltbest], roles=roles)
train = dataprovider.get_train_set()
val = dataprovider.get_val_set()
test = dataprovider.get_test_set()
quantiles = [0.1, 0.5, 0.9]
model_path = '/home/dior00002/dfki/forcateri/_data/dartstcn/dartstcn.pt'

print(test)

# dartstcn = DartsTCNModel(quantiles=quantiles)
# dartstcn.fit(train_data = train, val_data=val)



#Here prediction is darts timeseries
dartstcn = DartsTCNModel.load(model_path)
dartstcn.quantiles = quantiles
num_forecast_steps = 2
prediction = dartstcn.predict(
    data=test, # Pass the series used for prediction context
    predict_likelihood_parameters = True,
    n=num_forecast_steps,
)
prediction_ts = dartstcn.to_time_series(prediction)
print(prediction_ts[0].data)
#dartstcn.save(model_path)
#print(prediction)