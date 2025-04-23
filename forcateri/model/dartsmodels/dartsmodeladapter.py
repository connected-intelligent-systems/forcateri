from ..modelexceptions import  ModelAdapterError,InvalidModelTypeError
from ..modeladapter import ModelAdapter
from ...data.timeseries import TimeSeries
from typing import List,Optional, Any, Union, Tuple
import logging
from pathlib import Path
import os
import pandas as pd
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries as DartsTimeSeries


class DartsModelAdapter(ModelAdapter):
    def __init__(self, *args,**kwargs):
        super().__init__(*args,**kwargs)
        
    def fit(self,target:Union[TimeSeries, List[TimeSeries]] = None,
                 known:Union[TimeSeries, List[TimeSeries]] = None,
                 observed:Union[TimeSeries, List[TimeSeries]] = None,
                 static:Union[TimeSeries, List[TimeSeries]] = None,
                 time_col:str = 'time_stamp',
                 freq:str = '60min',**kwargs) -> None:
        super().fit(**kwargs)
        self.fit_args = {'target':self.to_model_format(ts = target)}
        if self.model.supports_future_covariates and known is not None:
            self.fit_args["future_covariates"] = self.to_model_format(ts = known,time_col=time_col,freq=freq)

        if self.model.supports_past_covariates and observed is not None:
            self.fit_args["past_covariates"] = self.to_model_format(ts = observed)
            
        if self.model.supports_static_covariates and static is not None:
            self.fit_args["static_covariates"] = self.to_model_format(ts = static)
    
    def flatten_timeseries_df(self,df: pd.DataFrame) -> pd.DataFrame:
            # Reset index to make 'time_stamp' a column
            df_reset = df.reset_index()

            # Drop the 'offset' column if it's not needed
            if 'offset' in df_reset.columns:
                df_reset = df_reset.drop(columns='offset')

            # Flatten the column MultiIndex
            df_reset.columns = [
                col if not isinstance(col, tuple) else col[0] for col in df_reset.columns
            ]

            # Ensure 'time_stamp' is the first column
            cols = df_reset.columns.tolist()
            if 'time_stamp' in cols:
                cols.insert(0, cols.pop(cols.index('time_stamp')))
                df_reset = df_reset[cols]

            

            return df_reset

    
    def to_model_format(self, ts: Union[TimeSeries, List[TimeSeries]],time_col:str = 'time_stamp',freq:str = '60min') -> Union[DartsTimeSeries, List[DartsTimeSeries]]:
        """
        Converts a TimeSeries object or a list of TimeSeries objects into a DartsTimeSeries object.
        This method takes a TimeSeries object or a list of TimeSeries objects, flattens the data, 
        and converts it into a DartsTimeSeries object. The conversion can handle grouped data if 
        a grouping column is specified.

        Args:
            ts (Union[TimeSeries, List[TimeSeries]]): A TimeSeries object or a list of TimeSeries 
                objects to be converted.

        Returns:
            DartsTimeSeries: A DartsTimeSeries object created from the input data.
        """
        def process_single_time_series(t: TimeSeries) -> DartsTimeSeries:
            data = self.flatten_timeseries_df(t)
            data[time_col] = pd.to_datetime(data[time_col]).dt.tz_localize(None)
            value_cols = [col for col in data.columns if col != time_col]
            return DartsTimeSeries.from_dataframe(data, time_col=time_col, value_cols= value_cols, freq=freq)

        if isinstance(ts, list):
            return [process_single_time_series(t) for t in ts]
        else:
            return process_single_time_series(ts)
        
    
    def split_series(self,
        ts: Union[DartsTimeSeries, List[DartsTimeSeries]],
        split: Union[float, pd.Timestamp]
    ) -> Union[
        Tuple[DartsTimeSeries, DartsTimeSeries],
        Tuple[List[DartsTimeSeries], List[DartsTimeSeries]]
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

        def _split_single(series: DartsTimeSeries):
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
        
    def to_time_series(ts:DartsTimeSeries):
        #Need to think of the way to implement this method.
        return super().to_time_series()
    
    def predict(self):
        raise NotImplementedError("Subclasses must implement this method.")
    def tune(self):
        raise NotImplementedError("Subclasses must implement this method.")
    
    def load(self, path: Union[Path, str]) -> None:
        """
        Loads a model from the specified path.

        Parameters
        ----------
        path : Union[Path, str]
            The file path to the saved model.

        Raises
        ------
        InvalidModelTypeError
            If the loaded model is not a valid Darts forecasting model.
        ModelAdapterError
            If the model fails to load due to file not found or other I/O errors.
        """
        try:
            model = ForecastingModel.load_model(path)
            if not isinstance(model, ForecastingModel):
                raise InvalidModelTypeError("The loaded model is not a valid Darts model.")
            else:
                self.model = model
                logging.info(f"Model loaded from {path}")
        except Exception as e:
            logging.error(f"Failed to load the model from {path}, check the model path")
            raise ModelAdapterError("Failed to load the model.") from e
        
        
    def save(self,path: Union[Path, str]) -> None:
        """
        Saves the model to the specified path.

        Parameters
        ----------
        path : Union[Path, str]
            The file path where the model will be saved.

        Raises
        ------
        ModelAdapterError
            If the model fails to save due to I/O errors.
        """
        try:
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            logging.info(f"Model saved to {path}")
        except Exception as e:
            logging.error(f"Failed to save the model to {path}")
            raise ModelAdapterError("Failed to save the model.") from e
    