from ..modeladapter import ModelAdapter, ModelAdapterError
from ...data.timeseries import TimeSeries
from typing import List,Optional, Any, Union, Tuple
import logging
from pathlib import Path
import pandas as pd
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries as darts_TimeSeries

class DartsModelAdapter(ModelAdapter):
    def __init__(self, model:ForecastingModel,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = model
        self.data = kwargs.get("data")
        #TODO think of how to make use of the ModelAdapter class variables, self.target, self.known, self.static
        self.fit_args = None
        
    def fit(self,**kwargs) -> None:
        super().fit(**kwargs)
        self.fit_args = {'target':self.to_model_format(ts = self.data,covariate_type='target',**kwargs)}
        if self.model.supports_future_covariates and kwargs.get("future_covariates"):
            self.fit_args["future_covariates"] = self.to_model_format(ts = self.data,covariate_type='future_cov',**kwargs)

        if self.model.supports_past_covariates and kwargs.get("past_covariates"):
            self.fit_args["past_covariates"] = self.to_model_format(ts = self.data,covariate_type='past_cov',**kwargs)
            
        if self.model.supports_static_covariates and kwargs.get("static_covariates"):
            self.fit_args["static_covariates"] = self.to_model_format(ts = self.data,covariate_type='static_cov',**kwargs)

        
    def to_model_format(self, covariate_type:str, **kwargs) -> Any:
            
        """
        Converts a TimeSeries to a Darts TimeSeries object.

        Parameters
        ----------
        ts : TimeSeries
            Input time series data.

        covariate_type : str
            One of 'target', 'past_cov', 'future_cov', or 'static_cov'.
            Determines which value_cols to use from kwargs.

        kwargs : dict
            Can include:
                - 'time_col': str
                - 'group_col': List[str] or str
                - 'freq': str
                - '{covariate_type}_value_cols': str or List[str]  (e.g., 'future_cov_value_cols')

        Returns
        -------
        darts.TimeSeries
        """
        time_col = kwargs.get("time_col")
        group_col = kwargs.get("group_col")
        freq = kwargs.get("freq")
        if covariate_type == 'target':
            value_cols = kwargs.get('target')
        else:
            value_cols = kwargs.get(f"{covariate_type}_value_cols")
        if group_col:
            darts_ts = darts_TimeSeries.from_group_dataframe(self.data,time_col=time_col,group_cols=group_col, value_cols=value_cols, freq=freq)
        else:
            darts_ts = darts_TimeSeries.from_dataframe(self.data,time_col=time_col,value_cols=value_cols, freq=freq)
        return darts_ts
    
    def split_series(self,
        ts: Union[darts_TimeSeries, List[darts_TimeSeries]],
        split: Union[float, pd.Timestamp]
    ) -> Union[
        Tuple[darts_TimeSeries, darts_TimeSeries],
        Tuple[List[darts_TimeSeries], List[darts_TimeSeries]]
    ]:
        """
        Splits Darts TimeSeries or list of TimeSeries into training and testing sets.

        Parameters
        ----------
        ts : DartsTimeSeries or List[DartsTimeSeries]
            The time series or list of time series to split.

        split : float or pd.Timestamp
            - If float (0 < split < 1), splits by percentage of the series length.
            - If pd.Timestamp, splits at that specific timestamp.

        Returns
        -------
        Tuple
            - Single TimeSeries: (train_ts, test_ts)
            - List of TimeSeries: (list of train_ts, list of test_ts)
        """

        def _split_single(series: darts_TimeSeries):
            if isinstance(split, float):
                if not 0 < split < 1:
                    raise ValueError("Split ratio must be between 0 and 1.")
                split_index = int(len(series) * split)
                return series[:split_index], series[split_index:]
            elif isinstance(split, pd.Timestamp):
                if split not in series.time_index:
                    raise ValueError(f"Timestamp {split} not in series index.")
                return series.split_before(split)
            else:
                raise TypeError("Split must be float or pd.Timestamp.")

        if isinstance(ts, list):
            return zip(*[_split_single(s) for s in ts])
        else:
            return _split_single(ts)
        
    def to_time_series(ts:darts_TimeSeries):
        #Need to think of the way to implement this method
        return super().to_time_series()
    
    def predict(self):
        raise NotImplementedError("Subclasses must implement this method.")
    def tune(self):
        raise NotImplementedError("Subclasses must implement this method.")
    def load(self):
        raise NotImplementedError("Subclasses must implement this method.")
    def save(self):
        raise NotImplementedError("Subclasses must implement this method.")
    