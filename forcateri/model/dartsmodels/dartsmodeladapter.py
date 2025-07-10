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
            if supports and value is not None:
                args[key] = value
        return args

    def fit(
        self, train_data: List[AdapterInput], val_data: Optional[List[AdapterInput]]
    ) -> None:
        """
        Fits the model using the provided training and validation data.
        """
        target, known, observed, static = self.convert_input(train_data)
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

    @abstractmethod
    def predict(
        self, data: List[AdapterInput], n: Optional[int] = 1
    ) -> List[DartsTimeSeries]:
        """
        Predict using the model and provided data.
        """
        target, known, observed, static = self.convert_input(data)
        predict_args = {"series": target}
        predict_args.update(self._get_covariate_args(known, observed, static))

        self._predict_args = predict_args
        self.last_time_stamps = [t.target.data.index[-1][1] for t in data]

        # return prediction

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

    def to_time_series(
        self, ts: Union[DartsTimeSeries, List[DartsTimeSeries]]
    ) -> Union[TimeSeries, List[TimeSeries]]:
        logger.info("Converting DartsTimeSeries to TimeSeries format.")
        if self.isquantile:
            if type(ts) == list:
                timeseries = []
                for id, darts_ts in enumerate(ts):
                    t0 = self.last_time_stamps[id]
                    darts_df = darts_ts.to_dataframe()
                    ts_obj = TimeSeries(
                        darts_df, representation="quantile", quantiles=self.quantiles
                    )
                    offset = (
                        ts_obj.data.index.get_level_values("time_stamp") - t0
                    )  # TODO Need to think about the logic of counting the offsets, and link to horizon of the dartsmodel
                    new_index = pd.MultiIndex.from_arrays(
                        [offset, [t0] * len(offset)],
                        names=["offset", "time_stamp"],
                    )
                    ts_obj.data.index = new_index
                    timeseries.append(ts_obj)
                return timeseries
            else:
                t0 = self.last_time_stamps[0]
                darts_df = ts.to_dataframe()
                ts_obj = TimeSeries(
                    darts_df, representation="quantile", quantiles=self.quantiles
                )
                offset = ts_obj.data.index.get_level_values("time_stamp") - t0
                new_index = pd.MultiIndex.from_arrays(
                    [offset, ts_obj.data.index.get_level_values("time_stamp")],
                    names=["offset", "time_stamp"],
                )
                ts_obj.data.index = new_index
                return ts_obj
        logger.info("Not a quantile prediction, converting to TimeSeries format.")
        raise NotImplementedError(
            "Conversion to TimeSeries format is not implemented for non-quantile predictions."
        )
        return ts

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
            # if not isinstance(model, ForecastingModel):
            #     raise InvalidModelTypeError(
            #         "The loaded model is not a valid Darts model."
            #     )
            # else:

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
