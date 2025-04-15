from .modeladapter import ModelAdapter, ModelAdapterError
from ..data.timeseries import TimeSeries
from typing import List,Optional, Any, Union
import logging
from pathlib import Path
import pandas as pd
from darts.models.forecasting.forecasting_model import ForecastingModel

class DartsModelAdapter(ModelAdapter):
    def __init__(self, model:ForecastingModel,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = model

    def fit(self,**kwargs) -> None:
        
        fit_args = {'series':self.target}
        super().fit(**kwargs)
        if self.model.supports_future_covariates and kwargs.get("future_covariates"):
            fit_args["future_covariates"] = self.to_model_format(kwargs["future_covariates"])

        if self.model.supports_past_covariates and kwargs.get("past_covariates"):
            fit_args["past_covariates"] = self.to_model_format(kwargs["past_covariates"])

        if self.model.supports_static_covariates and kwargs.get("static_covariates"):
            fit_args["static_covariates"] = self.to_model_format(kwargs["static_covariates"])
        try:
            self.model.fit(**fit_args)
        except ModelAdapterError as e:
            logging.error("Failed to fit a model, check the model params")
            raise ModelAdapterError(f"Failed to fit model: {e}") 
        
    def to_model_format(ts) -> Any:
        return super().to_model_format()
    
    def predict(self):
        raise NotImplementedError("Subclasses must implement this method.")
    def tune(self):
        raise NotImplementedError("Subclasses must implement this method.")
    def load(self):
        raise NotImplementedError("Subclasses must implement this method.")
    def save(self):
        raise NotImplementedError("Subclasses must implement this method.")