import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union


import pandas as pd
from darts import TimeSeries as DartsTimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.forecasting_model import ForecastingModel

from ...data.adapterinput import AdapterInput
from ...data.timeseries import TimeSeries
from ..modeladapter import ModelAdapter
from ..modelexceptions import InvalidModelTypeError, ModelAdapterError

logger = logging.getLogger(__name__)


class DartsModelAdapter(ModelAdapter, ABC):

    def __init__(self, freq: str = "60min", *args, **kwargs):
        self.freq = freq
        super().__init__(*args, **kwargs)
        self.model = None
        self.quantiles = kwargs.get("quantiles", None)

    def _get_covariate_args(self, known, observed, static):
        """
        Helper method to build covariate arguments for model fitting and prediction.
        """
        covariate_map = {
            "future_covariates": (
                getattr(self.model, "supports_future_covariates", False),
                known,
            ),
            "past_covariates": (
                getattr(self.model, "supports_past_covariates", False),
                observed,
            ),
            "static_covariates": (
                getattr(self.model, "supports_static_covariates", False),
                static,
            ),
        }
        args = {}
        for key, (supports, value) in covariate_map.items():
            if not supports or value is None:
                continue

            # if value is a list, skip if all elements are None or empty
            if isinstance(value, list):
                if all(v is None or (hasattr(v, '__len__') and len(v) == 0) for v in value):
                    continue
           
            
            args[key] = value
        
        return args

    def fit(
        self, train_data: List[AdapterInput], val_data: Optional[List[AdapterInput]]
    ) -> None:
        """
        Fits the model using the provided training and validation data.
        """
        target, known, observed, static = self.convert_input(train_data)
        print(static)
        fit_args = {"series": target}
        fit_args.update(self._get_covariate_args(known, observed, static))

        if val_data:
            val_target, val_known, val_observed, val_static = self.convert_input(
                val_data
            )
            fit_args["val_series"] = val_target
            val_covariate_args = self._get_covariate_args(
                val_known, val_observed, val_static
            )
            # Prefix validation covariate keys with 'val_'
            for key, value in val_covariate_args.items():
                fit_args[f"val_{key}"] = value

        self.model.fit(**fit_args)

    def prepare_predict_args(
        self, data: List[AdapterInput]
    ) -> Union[DartsTimeSeries, List[DartsTimeSeries]]:
        """
        Predict using the model and provided data.
        """
        target, known, observed, static = self.convert_input(data)
        predict_args = {"series": target}
        predict_args.update(self._get_covariate_args(known, observed, static))

        self._predict_args = predict_args

    @abstractmethod
    def predict(self, *args, **kwargs) -> Union[TimeSeries, List[TimeSeries]]:
        raise NotImplementedError(
            "The predict method is not implemented in the base DartsModelAdapter class. "
            "Please implement this method in the subclass."
        )

    @staticmethod
    def flatten_timeseries_df(df: pd.DataFrame) -> pd.DataFrame:
        # Sort index lexicographically to avoid PerformanceWarning
        df = df.sort_index(level=list(df.index.names), sort_remaining=True)
        df_reset = df.reset_index()

        # Drop the 'offset' column if it's not needed
        if "offset" in df_reset.columns:
            df_reset = df_reset.drop(columns="offset")

        # Flatten the column MultiIndex
        df_reset.columns = [
            col if not isinstance(col, tuple) else col[0] for col in df_reset.columns
        ]

        # Ensure 'time_stamp' is the first column
        cols = df_reset.columns.tolist()
        if "time_stamp" in cols:
            cols.insert(0, cols.pop(cols.index("time_stamp")))
            df_reset = df_reset[cols]

        return df_reset

    def to_model_format(self, t: TimeSeries) -> DartsTimeSeries:
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
        data["time_stamp"] = pd.to_datetime(data["time_stamp"]).dt.tz_localize(None)
        value_cols = [col for col in data.columns if col != "time_stamp"]
        return DartsTimeSeries.from_dataframe(
            data, time_col="time_stamp", value_cols=value_cols, freq=self.freq
        )

    def convert_input(self, input: List[AdapterInput]) -> Tuple[
        List[DartsTimeSeries],
        List[DartsTimeSeries],
        List[DartsTimeSeries],
        Optional[pd.DataFrame],
    ]:
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

        target = [self.to_model_format(t.target) for t in input]
        known = [self.to_model_format(t.known) for t in input]
        observed = [self.to_model_format(t.observed) for t in input]
        static = [t.static for t in input]

        return target, known, observed, static

    @staticmethod
    def to_time_series(
        ts: Union[DartsTimeSeries, List[DartsTimeSeries]],
        quantiles: Optional[List[float]] = None,
        freq: str = "h",
    ) -> TimeSeries:
        """
        Converts a DartsTimeSeries or a list of DartsTimeSeries into a pandas DataFrame (or list of DataFrames).
        """

        def convert_single_ts(darts_ts: DartsTimeSeries) -> TimeSeries:

            darts_df = darts_ts.to_dataframe()

            new_offsets = [
                pd.Timedelta(i, freq) for i in range(1, len(darts_df.index) + 1)
            ]
            adjusted_times = darts_df.index - pd.to_timedelta(new_offsets)
            darts_df.index = pd.MultiIndex.from_arrays(
                [new_offsets, adjusted_times],
                names=TimeSeries.ROW_INDEX_NAMES,
            )

            ts_obj = TimeSeries(
                data=darts_df,
                representation=TimeSeries.QUANTILE_REP,
                quantiles=quantiles,
            )
            return ts_obj

        if isinstance(ts, list):
            df = pd.concat(
                [convert_single_ts(darts_ts).data for darts_ts in ts], axis=0
            )
            return TimeSeries(
                data=df, representation=TimeSeries.QUANTILE_REP, quantiles=quantiles
            )

        else:
            return convert_single_ts(ts)

    @abstractmethod
    def tune(
        self,
        train_data: List[AdapterInput],
        val_data: Optional[List[AdapterInput]],
        **kwargs,
    ):
        raise NotImplementedError("Subclasses must implement this method.")

    @classmethod
    def load(cls, path: Union[Path, str]) -> "DartsModelAdapter":
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
            model = ForecastingModel.load(path)
            logging.info(f"Model loaded from {path}")
            return cls(model=model)
        except Exception as e:
            logging.error(f"Failed to load the model from {path}, check the model path")
            raise ModelAdapterError("Failed to load the model.") from e

    def save(self, path: Union[Path, str]) -> None:
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
