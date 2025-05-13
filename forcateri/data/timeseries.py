from __future__ import annotations

from datetime import datetime
import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TimeSeries:
    def __init__(
        self,
        data: pd.DataFrame,
        time_col: Optional[str] = None,
        value_cols: Optional[Union[List[str], str]] = None,
        **kwargs,
    ):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame")

        # If already in internal format (e.g. MultiIndex on both axes), just store it
        if self._is_internal_format(data):
            self.data = data.copy()
            logger.info("TimeSeries initialized from internal-format DataFrame.")
        else:
            logger.info("Raw DataFrame provided, converting to TimeSeries format.")
            self.data = self._build_internal_format(
                data, time_col, value_cols, **kwargs
            )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        time_col: Optional[str] = "time_stamp",
        value_cols: Optional[Union[List[str], str]] = None,
        freq: Optional[Union[str, int]] = "h",
        ts_type: Optional[str] = "determ",
    ) -> TimeSeries:
        logger.info("Creating TimeSeries from DataFrame via class method.")
        formatted = cls._build_internal_format(
            df, time_col, value_cols, freq=freq, ts_type=ts_type
        )
        return cls(formatted)

    @staticmethod
    def _is_internal_format(df: pd.DataFrame) -> bool:
        if not isinstance(df.index, pd.MultiIndex) and isinstance(
            df.columns, pd.MultiIndex
        ):
            return False
        expected_index_names = ['offset', 'time_stamp']
        expected_column_names = ['feature', 'representation']
        return df.index.names == expected_index_names and df.columns.names == expected_column_names

    @staticmethod
    def _build_internal_format(
        df: pd.DataFrame,
        time_col: Optional[str],
        value_cols: Optional[Union[List[str], str]],
        freq: Optional[Union[str, int]] = "h",
        ts_type: Optional[str] = "determ",
    ) -> pd.DataFrame:
        if not isinstance(time_col, str):
            raise TypeError("time_col must be a string.")
        if time_col not in df.columns:
            raise ValueError(f"Column {time_col} not found in DataFrame.")
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])

        t0_index = pd.date_range(
            start=df[time_col].min(), end=df[time_col].max(), freq=freq
        )
        if value_cols is None:
            value_cols = df.columns[df.columns != time_col]
        elif isinstance(value_cols, str):
            value_cols = [value_cols]
        features = value_cols

        row_dim_names = ["offset", "time_stamp"]
        col_dim_names = ["feature", "representation"]

        if ts_type == "determ":
            point_0_index = [pd.Timedelta(0)]
            point_0_row_index = pd.MultiIndex.from_product(
                [point_0_index, t0_index], names=row_dim_names
            )
            determ_col_index = pd.MultiIndex.from_product(
                [features, ["value"]], names=col_dim_names
            )
            df = df[features]
            return pd.DataFrame(
                df.values, index=point_0_row_index, columns=determ_col_index
            )

        elif ts_type == "sampled":
            sampled_cols = [f"s_{i}" for i in range(16)]
            point_1_row_index = pd.MultiIndex.from_product(
                [pd.to_timedelta([1], unit="h"), t0_index], names=row_dim_names
            )
            sampled_col_index = pd.MultiIndex.from_product(
                [features, sampled_cols], names=col_dim_names
            )
            return pd.DataFrame(
                df[features].values, index=point_1_row_index, columns=sampled_col_index
            )

        elif ts_type == "quantile":
            quant_cols = ["q_0.1", "q_0.5", "q_0.9"]
            range_index = pd.to_timedelta(np.arange(1, 25), unit="h")
            range_row_index = pd.MultiIndex.from_product(
                [range_index, t0_index], names=row_dim_names
            )
            quant_col_index = pd.MultiIndex.from_product(
                [features, quant_cols], names=col_dim_names
            )
            return pd.DataFrame(
                df[features].values, index=range_row_index, columns=quant_col_index
            )

        else:
            raise ValueError(
                "Invalid ts_type provided. Use 'determ', 'sampled', or 'quantile'."
            )

    @classmethod
    def from_group_df(
        cls,
        df: pd.DataFrame,
        group_col: str,
        time_col: Optional[str] = None,
        value_cols: Optional[Union[List[str], str]] = None,
        freq: Optional[Union[str, int]] = "h",
        ts_type: Optional[str] = "determ",
    ) -> List[pd.DataFrame]:
        """
        Build `TimeSeries` instances for each group in the DataFrame.

        This method groups the DataFrame by the specified `group_col` and applies the logic
        from `from_dataframe` to each group. Each group is expected to contain a time column (or index)
        and one or more value columns representing the time series data.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame from which to initialize the instances.
        group_col : str
            The column name used for grouping the data. Each unique value in this column will
            result in a separate `TimeSeries` instance.
        time_col : Optional[str], default None
            The name of the column in the DataFrame that contains time information.
            If provided, this column must exist in the DataFrame.
        value_cols : Optional[Union[List[str], str]], default None
            The name(s) of the column(s) in the DataFrame that contain the values.
            Can be a single column name or a list of column names.
        freq : Optional[Union[str, int]], default 'h'
            The frequency of the time series data.
        ts_type : Optional[str], default 'determ'
            The type of the time series to create. Use 'quantile' for quantile forecasts,
            'determ' for deterministic series, and 'sampled' for sampled series.

        Returns
        -------
        List[TimeSeries]
            A list of `TimeSeries` instances, one for each unique group in the DataFrame.

        Raises
        ------
        ValueError
            If `group_col` is not found in the DataFrame.
        """
        if group_col not in df.columns:
            logger.error("Initialization failed: group_col not found in the DataFrame.")
            raise ValueError(f"Column {group_col} not found in the DataFrame.")
        # if value_cols is None:
        #         value_cols = df.columns[df.columns != time_col]
        unique_group = df[group_col].unique()
        ts_dict = {}
        ts_list = []
        for i, group_id in enumerate(unique_group):
            df_group = df[df[group_col] == group_id]
            ts_instance = cls.from_dataframe(
                df_group, time_col, value_cols, freq, ts_type
            )
            ts_dict[group_id] = ts_instance
            ts_list.append(ts_instance)
            ts_dict[i] = group_id
        return ts_list, ts_dict

    def to_samples(self, n_samples: int) -> pd.DataFrame:
        """
        Generate Monte Carlo samples based on quantiles for each column.

        This method generates synthetic samples by applying Monte Carlo sampling
        to the empirical quantiles of each column in the time series DataFrame.
        The sampling is done independently for each feature using inverse transform sampling.

        Parameters
        ----------
        n_samples : int
            The number of samples to generate.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing `n_samples` rows, where each column is sampled
            based on the empirical quantile function of the corresponding feature.

        Raises
        ------
        ValueError
            If `n_samples` is not a positive integer.
        """
        # if n_samples <= 0:
        #     logger.error("n_samples is not positive integer")
        #     raise ValueError("n_samples must be a positive integer.")

        # probs = np.random.uniform(0, 1, (n_samples, self.data.shape[1]))  # Random probabilities
        # sampled_data = pd.DataFrame(
        #     {col: np.quantile(self.data[col], probs[:, i]) for i, col in enumerate(self.data.columns)},
        #     columns=self.data.columns
        # )
        # return sampled_data
        # TODO from quantiles to samples, This method is not really applicable
        raise NotImplementedError()

    def to_quantiles(self, quantiles: List[float] = [0.1, 0.5, 0.9]) -> pd.DataFrame:
        """
        Compute empirical quantiles from the time series data.

        This method calculates the specified quantiles for each column in the time series.
        The quantiles summarize the distribution of values and can be useful for probabilistic forecasting.

        Parameters
        ----------
        quantile_levels : list of float
            A list of quantile levels (between 0 and 1) to compute.
            Example: [0.1, 0.5, 0.9] for 10th, 50th (median), and 90th percentiles.

        Returns
        -------
        pd.DataFrame
            A DataFrame where each row corresponds to a quantile level and
            each column corresponds to a feature in the time series.

        Raises
        ------
        ValueError
            If any quantile level is not between 0 and 1.
        """

        # if not all(0 <= q <= 1 for q in quantiles):
        #     raise ValueError("Quantile levels must be between 0 and 1.")
        # quantile_values = self.data.quantile(quantiles)
        # return quantile_values
        raise NotImplementedError()

    def by_time(self, horizon: Optional[Union[int, pd.Timestamp]] = None):
        """
        Filter the data by a specific point in time or a time-based offset.

        Parameters
        ----------
        horizon : int or pd.Timestamp, optional
            - If a `pd.Timestamp` is provided, returns data corresponding to that exact timestamp.
            - If an `int` is provided, this can be used to implement logic such as selecting
            the last `n` time steps or indexing based on time step position (TODO).
            - If None or an unsupported type, a ValueError is raised.

        Returns
        -------
        pd.DataFrame
            A filtered subset of the original time series data based on the specified horizon.

        Raises
        ------
        ValueError
            If the provided `horizon` is neither a `pd.Timestamp` nor an `int`.

        Notes
        -----
        The DataFrame is assumed to have a MultiIndex where one of the levels is time-based.
        The `swaplevel(axis=0)` is used for convenient access by timestamp. Ensure that
        after swapping, timestamps are accessible at the top level of the index.
        """
        if isinstance(horizon, pd.Timestamp):
            return self.data.swaplevel(axis=0).loc(horizon)
        elif isinstance(horizon, int):
            # TODO the logic to handle int horizon
            raise NotImplementedError()
        else:
            logger.error("Incorrect format")
            raise ValueError("Please provide the pd.timestamp as horizon")

    def by_horizon(self, t0):
        """
        Return forecasts made at time `t0`, reindexed by their actual timestamps.

        Parameters
        ----------
        t0 : pd.Timestamp
            The time at which forecasts were made.

        Returns
        -------
        pd.DataFrame
            Forecasted values with the actual forecast time as the index.

        Raises
        ------
        KeyError
            If `t0` is not in the index.
        """
        try:
            forecasts = self.data.loc[t0]
            forecasts["time_stamp"] = forecasts.index + t0
            forecasts.set_index("time_stamp", inplace=True, drop=True)
            return forecasts
        except KeyError:
            logger.error(f"{t0} not found in forecast data.")
            raise ValueError(f"{t0} offset is not found in the forecast data")

    def split(self, timestamp):
        raise NotImplementedError()
        # TODO


    def get_feature_slice(self, index: List[str], copy: bool = False) -> TimeSeries:
        """
        Extracts a subset of the data based on the specified columns.
        Representations (level 1 of the column index) are carried over to the new `TimeSeries`.

        Parameters
        ----------
            features List[str]: The names of the features to keep in the returned `TimeSeries`
            copy bool, optional: Whether to copy the underlying data. Defaults to False.

        Returns
        -------
        TimeSeries
            A subseries containing a selection by features.

        Raises
        ------
        TypeError
            If index is not a List of strings
        """
        if (not isinstance(index, List)) or (
            not all([isinstance(i, str) for i in index])
        ):
            raise TypeError("features must be a list of strings")

        new_data = self.data[index]
        return TimeSeries(data=new_data.copy() if copy else new_data)

    def get_time_slice(self, index, copy: bool = False):
        """
        Extracts a subset of the data based on the specified time point or interval.
        Offsets (level 0 of the row index) are carried over to the new `TimeSeries`.

        Parameters
        ----------
            index: a single key or a slice indicating what part of the time series to access.
                - type int is interpreted as absolute number time steps.
                - type float is interpreted as relative offset based on the total number of time steps.
                - type datetime or pd.Timestamp is interpreted as point in time.
                - type slice slices the underlying data interpreting start and stop as one of the above types.
                    Start and stop do not need to be of the same type.
                    Step must be None.
            copy bool, optional: Whether to copy the underlying data. Defaults to False.

        Returns
        -------
        TimeSeries
            A subseries containing a selection by time.

        Raises
        ------
        TypeError
            If index has none of the above types
        NotImplementedError
            When the step property of a slice is not None.
        """

        # conversion of various formats into timestamps
        def to_dt(i) -> Optional[Union[pd.Timestamp, slice]]:
            match i:
                case None:
                    return None
                case slice():
                    if i.step is not None:
                        raise NotImplementedError(
                            "Only continuous slices are supported."
                        )
                    else:
                        return slice(to_dt(i.start), to_dt(i.stop))
                case pd.Timestamp():
                    return i
                case datetime():
                    return pd.Timestamp(i)
                case int():
                    return self.data.index.levels[1][i]
                case float():
                    return to_dt(int(np.round(len(self) * i)))
                case _:
                    raise TypeError(f"Key {i} has unexpected type {type(i)}.")

        index = to_dt(index)

        # adjust index format so that the data frame structure is preserved upon access
        if not isinstance(index, slice):
            index = [index]

        # slice the underlying data
        new_data = (
            self.data.swaplevel(axis=0)
            .sort_index()
            .loc[index]
            .swaplevel(axis=0)
            .sort_index()
        )
        return TimeSeries(data=new_data.copy() if copy else new_data)

    def __repr__(self):
        return f"TimeSeries(data={self.data})"

    def __len__(self) -> int:
        """
        Returns the length of the time series in time steps.
        The number of offsets does not count towards the length.

        Returns
        -------
        int
            The number of time steps in the series.
        """
        return len(self.data.index.levels[1])

    def __getitem__(self, index) -> TimeSeries:
        """
        Allows collection style access via a variety of keys, where single keys or slices
        split along the time axis and lists of strings split along the feature axis.
        Offsets and representations are carried over to the new `TimeSeries`.
        The newly created `TimeSeries` operates on the same underlying data.

        Parameters
        ----------
            index: a single key or a slice indicating what part of the time series to access
                - type int is interpreted as absolute number time steps
                - type float is interpreted as relative offset based on the total number of time steps
                - type datetime or pd.Timestamp is interpreted as point in time
                - type slice slices the underlying data interpreting start and stop as one of the above types
                - type List[str] is interpreted as subset of features to select for and is forwarded to get_features
        Returns
        -------
        TimeSeries
            A subseries containing a selection by time or by features.

        Raises
        ------
        TypeError
            If index has none of the above types
        NotImplementedError
            When the step property of a slice is not None.

        Examples
        --------
        >>> from datetime import datetime, timezone
        >>> ts = TimeSeries(some_data)
        >>>
        >>> print(ts[3])
        >>> print(ts[:3])
        >>> print(ts[3:6])
        >>> print(ts[0.1: -4])
        >>> print(ts[datetime(2000, 1, 1, tzinfo=timezone.utc):])
        """

        if isinstance(index, List):
            return self.get_feature_slice(index)
        else:
            return self.get_time_slice(index)
