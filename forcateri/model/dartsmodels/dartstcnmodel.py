from .dartsmodeladapter import DartsModelAdapter
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries as darts_TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TCNModel
import logging
from ..modeladapter import InvalidModelTypeError, ModelAdapterError

class DartsTCNModel(DartsModelAdapter):
    def __init__(self, model:TCNModel,*args,**kwargs):
        if not isinstance(model,TCNModel):
            logging.error("The DartsTCNModel accepts only TCN model from darts library")
            raise InvalidModelTypeError
        super().__init__(model,*args,**kwargs)

        self.scaler_target = Scaler()
        self.scaler_cov = Scaler()
    
    def fit(self,**kwargs):
        super().fit(**kwargs)
        
        scaled_target = self.scaler_target.fit_transform(self.fit_args['target'])
        scaled_cov = self.scaler_cov.fit_transform(self.fit_args['past_covariates'])
        train_series, val_series = self.split_series(scaled_target,split=0.8)
        train_past_cov,val_past_cov = self.split_series(scaled_cov,split=0.8)

        try:
            self.model.fit(
                series=train_series,
                past_covariates=train_past_cov,
                val_series=val_series, 
                val_past_covariates=val_past_cov,
                verbose=True
            )
        except ModelAdapterError as e:
            logging.error("Failed to fit a model, check the model params")
            raise ModelAdapterError(f"Failed to fit model: {e}") 
   
    def predict(self,**kwargs):
        pass 

    