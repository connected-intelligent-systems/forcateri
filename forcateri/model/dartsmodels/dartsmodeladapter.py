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
from ...data.adapterinput import AdapterInput
from abc import abstractmethod, ABC

class DartsModelAdapter(ModelAdapter, ABC):
    
    def __init__(self, freq:str = '60min',*args,**kwargs):
        self.freq = freq
        super().__init__(*args,**kwargs)
        self.model = None

    def fit(self,train_data:List[AdapterInput],val_data:Optional[List[AdapterInput]]) -> None:
        """
        Fits the model using the provided training and validation data.
        This method prepares the input data by converting it into the required format
        and passes it to the model's `fit` method. It supports handling target series,
        future covariates, past covariates, and static covariates, depending on the
        model's capabilities.
        Parameters:
            train_data (List[AdapterInput]): The training data containing target series
                and optional covariates (future, past, and static).
            val_data (Optional[List[AdapterInput]]): The validation data containing target
                series and optional covariates (future, past, and static). If not provided,
                validation is skipped.
        Raises:
            ValueError: If the input data is not in the expected format or if the model
                does not support a required covariate type.
        """
        
        target, known, observed, static = self.convert_input(train_data)

        fit_args = {'series':target}
        covariate_map = {
            'future_covariates': (self.model.supports_future_covariates, known),
            'past_covariates': (self.model.supports_past_covariates, observed),
            'static_covariates': (self.model.supports_static_covariates, static),
        }

        for key, (supports, value) in covariate_map.items():
            if supports and value is not None:
                fit_args[key] = value

        if val_data:
            val_target, val_known, val_observed, val_static = self.convert_input(val_data)
            fit_args['val_series'] = val_target

            val_covariate_map = {
                'val_future_covariates': (self.model.supports_future_covariates, val_known),
                'val_past_covariates': (self.model.supports_past_covariates, val_observed),
                'val_static_covariates': (self.model.supports_static_covariates, val_static),
            }

            for key, (supports, value) in val_covariate_map.items():
                if supports and value is not None:
                    fit_args[key] = value
        self.model.fit(**fit_args)

    @staticmethod
    def flatten_timeseries_df(df: pd.DataFrame) -> pd.DataFrame:
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
    
  
    def to_model_format(self,t: TimeSeries) -> DartsTimeSeries:
            """
            Converts a TimeSeries object into a DartsTimeSeries object.

            This method processes the input TimeSeries object by flattening its data,
            removing timezone information from the 'time_stamp' column, and identifying
            the value columns. It then creates and returns a DartsTimeSeries object
            using the processed data.

            Parameters:
                t (TimeSeries): The input TimeSeries object to be converted.

            Returns:
                DartsTimeSeries: The resulting DartsTimeSeries object.

            Raises:
                ValueError: If the input data is not in the expected format or if
                            required columns are missing.

            Notes:
                - The 'time_stamp' column in the input data is expected to contain
                  datetime values.
                - The method assumes that all columns except 'time_stamp' are value
                  columns.
            """
            data = DartsModelAdapter.flatten_timeseries_df(t.data)
            data['time_stamp'] = pd.to_datetime(data['time_stamp']).dt.tz_localize(None)
            value_cols = [col for col in data.columns if col != 'time_stamp']
            return DartsTimeSeries.from_dataframe(data, time_col='time_stamp', value_cols= value_cols, freq=self.freq)
    

    
    def convert_input(self,data:List[AdapterInput]) -> Tuple[List[DartsTimeSeries], List[DartsTimeSeries], List[DartsTimeSeries], Optional[pd.DataFrame]]:
        """
        Converts a list of AdapterInput objects into a tuple of lists formatted for the Darts model.
        Parameters:
            data (List[AdapterInput]): A list of AdapterInput objects containing the input data.
        Returns:
            Tuple[List[DartsTimeSeries], List[DartsTimeSeries], List[DartsTimeSeries], Optional[pd.DataFrame]]:
                - A list of DartsTimeSeries objects representing the target time series.
                - A list of DartsTimeSeries objects representing the known time series.
                - A list of DartsTimeSeries objects representing the observed time series.
                - An optional pandas DataFrame containing static data, if available.
        """

        target = [self.to_model_format(t.target) for t in data]
        known = [self.to_model_format(t.known) for t in data]
        observed = [self.to_model_format(t.observed) for t in data]
        static = [t.static for t in data]
        
        return target, known, observed , static     
    
        
    def to_time_series(ts:DartsTimeSeries):
        #Need to think of the way to implement this method.
        return super().to_time_series()
    
    @abstractmethod
    def predict(self, data:List[AdapterInput]):
        raise NotImplementedError("Subclasses must implement this method.")
    
    @abstractmethod
    def tune(self, data:List[AdapterInput]):
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
    