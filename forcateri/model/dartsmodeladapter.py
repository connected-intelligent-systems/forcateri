import logging
import os
from abc import ABC
from pathlib import Path
from typing import List, Optional, Tuple, Union, Any, Dict


import pandas as pd
from darts import TimeSeries as DartsTimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.dataprocessing.transformers import Scaler

from ..data.adapterinput import AdapterInput
from ..data.timeseries import TimeSeries
from .modeladapter import ModelAdapter
from .modelexceptions import ModelAdapterError

logger = logging.getLogger(__name__)


class DartsModelAdapter(ModelAdapter, ABC):

    def __init__(
        self,
        freq: str = "60min",
        model_name: Optional[str] = None,
        quantiles: Optional[List[float]] = None,
        is_likelihood: bool = False,
        num_samples: Optional[int] = None,
    ):
        super().__init__(model_name=model_name)
        self.freq = freq
        self.model = None
        self.quantiles = quantiles
        self.scaler_target = None
        self.scaler_known = None
        self.scaler_observed = None
        self.is_likelihood = is_likelihood
        self.num_samples = num_samples
        self.scaler_target: Optional[Scaler] = None
        self.scaler_known: Optional[Scaler] = None
        self.scaler_observed: Optional[Scaler] = None

    def _get_covariate_args(self, known, observed):
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
            # "static_covariates": (
            #     getattr(self.model, "supports_static_covariates", False),
            #     static,
            # ),
        }
        args = {key: None for key in covariate_map}
        for key, (supports, value) in covariate_map.items():
            if not supports or value is None:
                logger.warning(
                    f"Model does not support {key} or no {key} provided, skipping this covariate."
                )
                continue

            # if value is a list, skip if all elements are None or empty
            if isinstance(value, list):
                if all(
                    v is None or (hasattr(v, "__len__") and len(v) == 0) for v in value
                ):
                    logger.warning(
                        f"All elements in {key} are None or empty, skipping this covariate."
                    )
                    continue

            args[key] = value

        return args['future_covariates'], args['past_covariates'] #, args['static_covariates']

    def fit(
        self,
        train_data: List[AdapterInput],
        val_data: Optional[List[AdapterInput]] = None,
    ) -> None:
        """
        Fits the Darts forecasting model using the provided training and validation data.

        This method converts the input data to Darts format, prepares the training arguments
        including any supported covariates (future, past, or static), and optionally includes
        validation data with the appropriate prefixes. The model is then fitted using these
        prepared arguments.

        Parameters
        ----------
        train_data : List[AdapterInput]
            A list of AdapterInput objects containing the training data, including target
            series and any available covariates (known/future, observed/past, and static).
        val_data : Optional[List[AdapterInput]], default=None
            An optional list of AdapterInput objects containing validation data. If provided,
            validation series and covariates will be passed to the model's fit method with
            'val_' prefixes.

        Returns
        -------
        None
            This method modifies the model in-place and does not return a value.

        Notes
        -----
        - The method automatically handles covariate support detection based on the model's
          capabilities (supports_future_covariates, supports_past_covariates, etc.).
        - Target column names are stored in self.target_col_names for later use in predictions.
        - If scalers are configured, they will be applied during the convert_input step.
        - Validation covariates are automatically prefixed with 'val_' to match Darts API
          requirements.
        """
        logger.debug(f"Starting model fit for {self.model_name}")
        target, known, observed, static = self.convert_input(train_data)
        self.target_col_names = [t.components[0] for t in target]
        logger.debug(f"Converted training data to darts format for {self.model_name}")


        future_covariates, past_covariates = self._get_covariate_args(known, observed)

        val_target, val_future_covariate, val_past_covariates = None, None, None
        if val_data is not None:
            val_target, val_known, val_observed, val_static = self.convert_input(
                val_data
            )

            logger.debug(
                f"Converted validation data to darts format for {self.model_name}"
            )
            
            val_future_covariate, val_past_covariates = self._get_covariate_args(
                val_known, val_observed
            )


        
        self.model.fit(series=target, 
                       future_covariates=future_covariates, 
                       past_covariates=past_covariates, 
                       val_series=val_target, 
                       val_future_covariates=val_future_covariate, 
                       val_past_covariates=val_past_covariates)

    def predict(
        self,
        data: List[AdapterInput],
        n: Optional[int] = 1,
        use_rolling_window: bool = True,
    ) -> List[TimeSeries]:
        """
        Generates predictions using the fitted Darts forecasting model.

        This method converts the input data to Darts format, prepares prediction arguments
        including covariates, and generates forecasts. It supports two prediction modes:
        rolling window (historical forecasts) for backtesting, or direct n-step ahead
        predictions.

        Parameters
        ----------
        data : List[AdapterInput]
            A list of AdapterInput objects containing the data for prediction, including
            target series and any available covariates (known/future, observed/past, and static).
        n : Optional[int], default=1
            The number of time steps ahead to forecast. Only used when rolling_window=False.
            For rolling window predictions, use the 'forecast_horizon' kwarg instead.
        rolling_window : bool, default=True
            If True, uses historical_forecasts method to generate multiple forecasts with
            a rolling window approach (useful for backtesting). If False, generates a single
            n-step ahead forecast from the end of the series.
        **kwargs
            Additional keyword arguments passed to the underlying prediction method:
            - For rolling_window=True: passed to historical_forecasts (e.g., forecast_horizon, stride)
            - For rolling_window=False: passed to model.predict (e.g., num_samples, mc_dropout)

        Returns
        -------
        List[TimeSeries]
            A list of TimeSeries objects containing the predictions. The format depends on
            the model type and configuration:
            - For probabilistic models: quantile or sample-based representations
            - For deterministic models: point forecasts
            - If scalers are configured, predictions are inverse-transformed to original scale

        Notes
        -----
        - The method automatically handles covariate preparation based on model capabilities.
        - For rolling window predictions, the forecast_horizon parameter in kwargs controls
          the prediction horizon at each step.
        - Predictions are automatically converted from Darts format back to the custom
          TimeSeries format with proper offset and time indexing.
        """
        target, known, observed, static = self.convert_input(data)
        #predict_args = self._prepare_predict_args(target, known, observed, static)
        future_covariates, past_covariates = self._get_covariate_args(known, observed)
        #print(f"Predict args: {predict_args.keys()}")
        if use_rolling_window:
            logger.debug("Using rolling window prediction.")
            return self._historical_forecasts(n=n, series=target, future_covariates=future_covariates, past_covariates=past_covariates)
        else:
            preds = self.model.predict(n=n, future_covariates=future_covariates, past_covariates=past_covariates)
            return self.convert_output(
                output=preds
            )  

    def convert_input(self, input):
        """
        Converts input data to Darts format and applies scaling transformations.

        This method extends the parent class's convert_input method by adding optional
        scaling transformations to the target series and covariates. Scalers are applied
        if they were configured during initialization.

        Parameters
        ----------
        input : List[AdapterInput]
            A list of AdapterInput objects containing the input data with target series,
            known/future covariates, observed/past covariates, and static covariates.

        Returns
        -------
        tuple
            A tuple containing four elements:
            - target : List[DartsTimeSeries] or DartsTimeSeries
                The target series, scaled if scaler_target is configured.
            - known : List[DartsTimeSeries] or DartsTimeSeries or None
                The known/future covariates, scaled if scaler_known is configured.
            - observed : List[DartsTimeSeries] or DartsTimeSeries or None
                The observed/past covariates, scaled if scaler_observed is configured.
            - static : pd.DataFrame or None
                The static covariates (not scaled).

        Notes
        -----
        - Scaling is only applied if the corresponding scaler was fitted during
          initialization (when scaler_data was provided).
        - The parent class's convert_input method handles the conversion from
          TimeSeries format to Darts format.
        - Scalers transform data to have zero mean and unit variance by default.
        """
        target, known, observed, static = super().convert_input(input)
        self.scaler_target = Scaler().fit(target)
        self.scaler_known = Scaler().fit(known) if known is not None else None
        self.scaler_observed = Scaler().fit(observed) if observed is not None else None
        logger.debug("Applying target scaler to target data.")
        target = self.scaler_target.transform(target)
        if self.scaler_known:
            logger.debug("Applying known scaler to known data.")
            known = self.scaler_known.transform(known)
        if self.scaler_observed:
            logger.debug("Applying observed scaler to observed data.")
            observed = self.scaler_observed.transform(observed)
        return target, known, observed, static

    def convert_output(
        self,
        output: Union[List[DartsTimeSeries], List[List[DartsTimeSeries]]],
        # is_likelihood: bool,
        # num_samples: Optional[int] = None,
    ) -> List[TimeSeries]:
        """
        Converts Darts model output to custom TimeSeries format with proper column naming.

        This method takes the raw output from a Darts forecasting model and converts it
        to the custom TimeSeries format used by the adapter. It handles both single and
        multiple series predictions, and properly formats column names to match the original
        target series names with their associated quantiles or samples.

        Parameters
        ----------
        output : Union[List[DartsTimeSeries], List[List[DartsTimeSeries]]]
            The raw output from the Darts model. Can be:
            - A list of DartsTimeSeries (for multiple predictions)
            - A single DartsTimeSeries (for single prediction)
            - A list of lists of DartsTimeSeries (for nested predictions)

        Returns
        -------
        List[TimeSeries]
            A list of TimeSeries objects with:
            - Proper offset and time indexing
            - MultiIndex columns with (feature, representation) structure
            - Original target column names restored
            - Appropriate representation type (quantile, sample, or deterministic)

        Notes
        -----
        - The method uses self.quantiles, self.is_likelihood, and self.num_samples
          to determine the output format.
        - For list outputs, each prediction is converted individually and column names
          are updated to match the original target column names stored in
          self.target_col_names.
        - Returns an empty list if the output is None or empty.
        - The conversion process includes proper handling of:
          * Quantile forecasts (for probabilistic models)
          * Sample-based forecasts (for stochastic models)
          * Point forecasts (for deterministic models)
        """
        if not output:
            return []
        if isinstance(output, list):
            # If the output is a list of lists, flatten it
            logger.debug("Converting list of DartsTimeSeries to TimeSeries format.")
            prediction_ts_format = [
                DartsModelAdapter.to_time_series(
                    ts=pred,
                    quantiles=self.quantiles,
                    is_likelihood=self.is_likelihood,
                    num_samples=self.num_samples,
                )
                for pred in output
            ]
            for ts, new_name in zip(prediction_ts_format, self.target_col_names):
                ts.data.columns = pd.MultiIndex.from_tuples(
                    [(new_name, q) for q in ts.data.columns.get_level_values(1)],
                    names=TimeSeries.COL_INDEX_NAMES,
                )
                logger.debug(
                    f"Renamed column names to match TimeSeries format: {ts.data.columns}"
                )
        else:
            logger.debug("Converting single DartsTimeSeries to TimeSeries format.")
            prediction_ts_format = DartsModelAdapter.to_time_series(
                ts=output,
                quantiles=self.quantiles,
                is_likelihood=self.is_likelihood,
                num_samples=self.num_samples,
            )

        return prediction_ts_format

    def _historical_forecasts(
        self,
        series: DartsTimeSeries,
        future_covariates: Optional[DartsTimeSeries] = None,
        past_covariates: Optional[DartsTimeSeries] = None,
        retrain: bool = False,
        n: Optional[int] = 1,
    ) -> List[TimeSeries]:
        """
        Generates historical forecasts using the provided data.

        Parameters:
            series (DartsTimeSeries): The input time series for generating forecasts.
            future_covariates (Optional[DartsTimeSeries]): Future covariates for the model.
            past_covariates (Optional[DartsTimeSeries]): Past covariates for the model.
            static_covariates (Optional[DartsTimeSeries]): Static covariates for the model.
            retrain (bool): Whether to retrain the model before each forecast.
            n (Optional[int]): The forecast horizon for historical forecasts.
        Returns:
            List[TimeSeries]: A list of TimeSeries objects representing the forecasts.
        """
        # target, known, observed, static = self.convert_input(data)
        # self._prepare_predict_args(target, known, observed, static)
        logger.debug("Generating historical forecasts.")
        preds = self.model.historical_forecasts(
            series=series,
            future_covariates=future_covariates,
            past_covariates=past_covariates,
            retrain=retrain,
            predict_likelihood_parameters=self.is_likelihood,
            forecast_horizon=n,
            last_points_only=False,
        )
        if self.scaler_target:
            logger.debug("Inverse transforming forecasts using target scaler.")
            preds = self.scaler_target.inverse_transform(preds)

        return self.convert_output(preds)

    @staticmethod
    def flatten_timeseries_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Flattens a TimeSeries DataFrame by resetting its MultiIndex and simplifying column structure.

        This static method prepares a TimeSeries DataFrame for conversion to Darts format by:
        - Sorting the index to avoid performance warnings
        - Resetting the MultiIndex to regular columns
        - Removing the 'offset' column (not needed for Darts)
        - Flattening MultiIndex columns to single-level columns
        - Ensuring 'time' is the first column

        Parameters
        ----------
        df : pd.DataFrame
            A TimeSeries DataFrame with MultiIndex rows (offset, time) and potentially
            MultiIndex columns (feature, representation).

        Returns
        -------
        pd.DataFrame
            A flattened DataFrame with:
            - Regular (non-MultiIndex) row index
            - 'time' as the first column
            - Single-level column names
            - No 'offset' column

        Notes
        -----
        - The input DataFrame is sorted lexicographically to avoid pandas PerformanceWarning.
        - MultiIndex columns are flattened by taking the first level (feature name).
        - The 'offset' column is dropped as Darts uses time directly for indexing.
        - This method is typically called as part of the to_model_format conversion pipeline.

        Examples
        --------
        Input DataFrame (MultiIndex):
            Index: [(0 days, '2020-01-01'), (1 day, '2020-01-02'), ...]
            Columns: [('feature1', 'value'), ('feature2', 'value')]

        Output DataFrame:
            Index: [0, 1, ...]
            Columns: ['time', 'feature1', 'feature2']
        """
        # Sort index lexicographically to avoid PerformanceWarning
        df = df.sort_index(level=list(df.index.names), sort_remaining=True)
        df_reset = df.reset_index()
        logger.debug(f"Flattened DataFrame columns: {df_reset.columns.tolist()}")
        # Drop the 'offset' column if it's not needed
        if "offset" in df_reset.columns:
            df_reset = df_reset.drop(columns="offset")
            logger.debug(
                "Dropped 'offset' column from DataFrame as part of to_model_format in dartsmodeladapter"
            )
        # Flatten the column MultiIndex
        df_reset.columns = [
            col if not isinstance(col, tuple) else col[0] for col in df_reset.columns
        ]

        # Ensure 'time' is the first column
        cols = df_reset.columns.tolist()
        if "time" in cols:
            cols.insert(0, cols.pop(cols.index("time")))
            df_reset = df_reset[cols]

        return df_reset

    def to_model_format(self, t: TimeSeries) -> DartsTimeSeries:
        """
        Converts a TimeSeries object into a DartsTimeSeries object.

        This method processes the input TimeSeries object by flattening its data,
        removing timezone information from the 'time' column, and identifying
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
            - The 'time' column in the input data is expected to contain
              datetime values.
            - The method assumes that all columns except 'time' are value
              columns.
        """
        data = DartsModelAdapter.flatten_timeseries_df(t.data)
        logger.debug(f"Data after flattening in to_model_format: {data.head()}")
        data["time"] = pd.to_datetime(data["time"]).dt.tz_localize(None)
        value_cols = [col for col in data.columns if col != "time"]
        return DartsTimeSeries.from_dataframe(
            data, time_col="time", value_cols=value_cols, freq=self.freq
        )

    @staticmethod
    def to_time_series(
        ts: Union[DartsTimeSeries, List[DartsTimeSeries]],
        quantiles: Optional[List[float]] = None,
        is_likelihood: bool = False,
        num_samples: Optional[int] = None,
        freq: str = "h",
    ) -> TimeSeries:
        """
        Converts DartsTimeSeries object(s) to forcateri TimeSeries format with proper indexing.

        This static method converts Darts forecasting model output into the custom TimeSeries
        format used by the adapter. It handles the conversion of time indices to offset-based
        indexing and properly sets the representation type (quantile, sample, or deterministic)
        based on the model configuration.

        Parameters
        ----------
        ts : Union[DartsTimeSeries, List[DartsTimeSeries]]
            A single DartsTimeSeries or a list of DartsTimeSeries objects to convert.
            Multiple series are concatenated into a single TimeSeries object.
        quantiles : Optional[List[float]], default=None
            List of quantile levels (e.g., [0.1, 0.5, 0.9]) if the predictions are
            probabilistic forecasts. If None, the output is treated as deterministic.
        is_likelihood : bool, default=False
            If True and quantiles are provided, indicates that the model predicted
            likelihood parameters for quantile regression. Determines the representation
            type of the output TimeSeries.
        num_samples : Optional[int], default=None
            Number of stochastic samples if the predictions are sample-based (e.g., from
            Monte Carlo dropout). If greater than 1, uses SAMPLE_REP representation.
        freq : str, default="h"
            Frequency string for the time deltas used in offset calculation.
            Examples: 'h' (hour), 'd' (day), 'min' (minute).

        Returns
        -------
        TimeSeries
            A TimeSeries object with:
            - MultiIndex rows: (offset, time)
              * offset: pd.Timedelta representing forecast horizon
              * time: actual time adjusted by subtracting offset
            - Appropriate representation type:
              * QUANTILE_REP if is_likelihood=True and quantiles provided
              * SAMPLE_REP if num_samples > 1
              * DETERM_REP otherwise

        Notes
        -----
        - The offset represents the forecast horizon (e.g., 1 hour ahead, 2 hours ahead).
        - Adjusted times are calculated as: original_time - offset, allowing reconstruction
          of the forecast origin time.
        - For list inputs, all series are concatenated along axis 0, preserving the
          MultiIndex structure.
        - The conversion handles three types of forecasts:
          1. Quantile forecasts (probabilistic with quantile levels)
          2. Sample-based forecasts (stochastic with multiple samples)
          3. Deterministic forecasts (point predictions)

        Examples
        --------
        Converting a single probabilistic forecast:
        >>> ts = to_time_series(
        ...     darts_ts,
        ...     quantiles=[0.1, 0.5, 0.9],
        ...     is_likelihood=True,
        ...     freq='h'
        ... )

        Converting multiple deterministic forecasts:
        >>> ts = to_time_series(
        ...     [darts_ts1, darts_ts2, darts_ts3],
        ...     freq='d'
        ... )
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
            if quantiles:
                if is_likelihood:
                    logger.debug(
                        "Converting single DartsTimeSeries with quantiles to TimeSeries"
                    )
                    ts_obj = TimeSeries(
                        data=darts_df,
                        representation=TimeSeries.QUANTILE_REP,
                        quantiles=quantiles,
                        freq=freq,
                    )
                elif num_samples is not None and num_samples > 1:
                    logger.debug(
                        "Converting single DartsTimeSeries with samples to TimeSeries"
                    )
                    ts_obj = TimeSeries(
                        data=darts_df,
                        representation=TimeSeries.SAMPLE_REP,
                        num_samples=num_samples,
                        freq=freq,
                    )
                else:
                    logger.debug(
                        "Converting single DartsTimeSeries from probabilist model with deterministic forecasts to TimeSeries"
                    )
                    ts_obj = TimeSeries(
                        data=darts_df, representation=TimeSeries.DETERM_REP, freq=freq
                    )
            else:
                logger.debug(
                    "Converting single DartsTimeSeries with deterministic forecasts to TimeSeries"
                )
                ts_obj = TimeSeries(
                    data=darts_df, representation=TimeSeries.DETERM_REP, freq=freq
                )
            return ts_obj

        if isinstance(ts, list):
            logger.debug("Converting list of DartsTimeSeries to TimeSeries")
            df = pd.concat(
                [convert_single_ts(darts_ts).data for darts_ts in ts], axis=0
            )
            if is_likelihood:
                return TimeSeries(
                    data=df, representation=TimeSeries.QUANTILE_REP, quantiles=quantiles
                )
            else:
                return TimeSeries(data=df, representation=TimeSeries.DETERM_REP)
        else:
            logger.debug("Converting single DartsTimeSeries to TimeSeries")
            return convert_single_ts(ts)

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
