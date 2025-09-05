from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional, Union, Tuple, Callable
from typing_extensions import Self

import numpy as np
import pandas as pd
from .timeseriesexceptions import InvalidDataFrameFormat, InvalidRepresentationFormat

logger = logging.getLogger(__name__)


class TimeSeries:

    DETERM_REP = "determ"
    QUANTILE_REP = "quantile"
    SAMPLE_REP = "sample"
    ROW_INDEX_NAMES: Tuple[str, str] = ("offset", "time_stamp")
    COL_INDEX_NAMES: Tuple[str, str] = ("feature", "representation")

    def __init__(
        self,
        data: pd.DataFrame,
        representation=None,
        quantiles: Optional[List[float]] = None,
        freq: Optional[str] = None,
    ):
        self._features = []
        self._representations = []
        self._offsets = pd.TimedeltaIndex([])
        self._timestamps = pd.DatetimeIndex([])

        if representation is None:
            if quantiles is None:
                representation = TimeSeries.DETERM_REP
            else:
                representation = TimeSeries.QUANTILE_REP

        self.quantiles = None
        self.representation = representation
        if representation == TimeSeries.QUANTILE_REP:
            if not all(isinstance(x, float) for x in quantiles):
                raise TypeError("Quantiles must be a list of floats.")
            if not all(0 <= x <= 1 for x in quantiles):
                raise ValueError("Quantiles must be between 0 and 1.")
            self.quantiles = quantiles
        elif representation == TimeSeries.SAMPLE_REP:
            # raise NotImplementedError("Sample representation is not implemented yet.")
            pass
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame")

        # If already in internal format (e.g. MultiIndex on both axes), just store it
        if TimeSeries.is_matching_format(data, self.representation):
            self.data = data.copy()

            logger.info("TimeSeries initialized from internal-format DataFrame.")
        elif TimeSeries.is_compatible_format(data, self.representation):
            # If the DataFrame is compatible but not in the expected format, align it
            self.data = data.copy()
            self.align_format(self.data)
            logger.info("TimeSeries initialized from compatible-format DataFrame.")
        else:
            logger.info(f"Raw DataFrame with {representation} cannot be aligned")
            raise InvalidDataFrameFormat(
                f"Cannot build TimeSeries from the provided DataFrame: "
                f"DataFrame is not in a matching or compatible format for representation '{representation}'. "
                f"Expected MultiIndex with index names {['offset', 'time_stamp']} and column names {['feature', 'representation']}."
                f"Or at least df with datetime index."
            )
        self._representations = list(
            self.data.columns.get_level_values(TimeSeries.COL_INDEX_NAMES[1]).unique()
        )
        self._features = list(
            self.data.columns.get_level_values(TimeSeries.COL_INDEX_NAMES[0]).unique()
        )
        self._offsets = self.data.index.get_level_values(
            TimeSeries.ROW_INDEX_NAMES[0]
        ).unique()
        self._timestamps = self.data.index.get_level_values(
            TimeSeries.ROW_INDEX_NAMES[1]
        ).unique()
        self._check_freq_format(self.data.index.get_level_values(0) + self.data.index.get_level_values(1), freq)

    @property
    def features(self):
        "The features property"
        return self._features

    @property
    def representations(self):
        "The representation property"
        return self._representations

    @property
    def offsets(self):
        "The offsets property"
        return self._offsets

    @property
    def timestamps(self):
        "The timestamps property"
        return self._timestamps

    def _check_freq_format(self, index: pd.Index, freq: Optional[str] = None) -> None:
        """
        Check and validate the frequency of a pandas DatetimeIndex.

        This method attempts to infer the frequency of the index using pandas'
        built-in `infer_freq`. If that fails, it manually computes the most
        common difference between consecutive timestamps. It then validates the
        inferred frequency against a user-provided frequency, if given.

        Parameters
        ----------
        index : pd.Index
            A pandas DatetimeIndex whose frequency needs to be checked.
        freq : str, optional
            Expected frequency string (e.g., 'D', 'H', '15min'). If provided,
            the method validates the index against this frequency.

        Raises
        ------
        TypeError
            If the index is not a pandas DatetimeIndex.
        ValueError
            If the inferred frequency does not match the provided `freq` or if
            the frequency cannot be inferred.
        """
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("Index must be a pandas.DatetimeIndex")

        logger.info("Checking the frequency format of the DataFrame")

        # Try pandas frequency inference first
        inferred_freq = pd.infer_freq(index)

        # Frequency inference logic if pandas fails
        if inferred_freq is None:
            logger.info("Pandas infer_freq failed, inferring manually")
            diffs = index[1:] - index[:-1]
            # Filter out zero or negative deltas (duplicates or non-monotonic index)
            diffs = diffs[diffs > pd.Timedelta(0)]

            if len(diffs) > 0:
                most_common_delta = diffs.value_counts().idxmax()
                logger.info(
                    f"Most common delta between timestamps: {most_common_delta}"
                )
                try:
                    inferred_freq = pd.tseries.frequencies.to_offset(
                        most_common_delta
                    ).freqstr
                    logger.info(f"Inferred frequency manually: {inferred_freq}")
                except ValueError:
                    logger.warning(
                        "Could not convert most common delta to frequency string"
                    )
                    inferred_freq = None

        # Validation logic
        if freq:
            logger.info(f"Validating provided frequency: {freq}")
            if inferred_freq and freq != inferred_freq:
                raise ValueError(
                    f"Provided freq {freq} is different from inferred freq {inferred_freq}"
                )
            self.freq = freq
        else:
            if inferred_freq is None:
                raise ValueError("Could not infer the frequency from the data")
            logger.info(f"Frequency set to inferred value: {inferred_freq}")
            self.freq = inferred_freq

    @staticmethod
    def _check_column_levels(
        df: pd.DataFrame, representation, strict: bool = False
    ) -> bool:
        match representation:
            case TimeSeries.DETERM_REP:
                logger.info(
                    "Checking DataFrame for deterministic representation. Feature levels must be unique."
                )
                feature_level_is_unique = df.columns.get_level_values(
                    TimeSeries.COL_INDEX_NAMES[0]
                ).nunique() == len(
                    df.columns.get_level_values(TimeSeries.COL_INDEX_NAMES[0])
                )
                level_value_is_unique = len(set(df.columns.get_level_values(1))) == 1
                if level_value_is_unique:
                    level_value_name_correct = (
                        set(df.columns.get_level_values(1)).pop() == "value"
                    )
                    return feature_level_is_unique and level_value_name_correct
                return False

            case TimeSeries.QUANTILE_REP:
                logger.info(
                    "Checking DataFrame for quantile representation. Quantile levels should match accross all features. Also, the levels should be floats."
                )
                matching_quantile_levels = (
                    df.columns.to_frame(index=False)
                    .groupby(TimeSeries.COL_INDEX_NAMES[0])[
                        TimeSeries.COL_INDEX_NAMES[1]
                    ]
                    .nunique()
                    .nunique()
                    == 1
                )
                if strict:
                    quantile_level_correct_name = all(
                        isinstance(x, float) for x in df.columns.get_level_values(1)
                    )
                    return matching_quantile_levels and quantile_level_correct_name
                else:
                    # If strict is False, we only check that quantile levels are unique across features
                    return matching_quantile_levels
            case TimeSeries.SAMPLE_REP:
                logger.info(
                    "Checking DataFrame for sample representation. Sample levels should be integers."
                )
                sample_level_correct_name = all(
                    isinstance(x, int) for x in df.columns.get_level_values(1)
                )
                return sample_level_correct_name
        return False

    @staticmethod
    def is_matching_format(df: pd.DataFrame, representation=None) -> bool:
        """
        Checks the structure of the row and the column index and returns true if a data frame
        has the expected format to serve as a TimeSeries data representation.
        No changes to the data or TimeSeries are made here.
        """
        if representation is None:
            representation = TimeSeries.DETERM_REP
        elif representation not in (
            TimeSeries.DETERM_REP,
            TimeSeries.QUANTILE_REP,
            TimeSeries.SAMPLE_REP,
        ):
            raise ValueError("Representation is not in required format")
        if isinstance(df.index, pd.MultiIndex) and isinstance(
            df.columns, pd.MultiIndex
        ):
            # Ensure index/column names match expected, and that the second level of index is a DatetimeIndex
            index_names_match = df.index.names == TimeSeries.ROW_INDEX_NAMES
            column_names_match = df.columns.names == TimeSeries.COL_INDEX_NAMES
            first_level_is_timedelta = isinstance(
                df.index.get_level_values(0), pd.TimedeltaIndex
            )
            second_level_is_datetime = isinstance(
                df.index.get_level_values(1), pd.DatetimeIndex
            )

            if (
                index_names_match
                and column_names_match
                and first_level_is_timedelta
                and second_level_is_datetime
            ):
                # Check specific representation requirements
                return TimeSeries._check_column_levels(df, representation, strict=True)
        return False

    @staticmethod
    def is_compatible_format(df: pd.DataFrame, representation) -> bool:
        """
        Checks the structure of the row and the column index and returns true if all the missing
        and or mislabeled information can be inferred.
        No changes to the data or TimeSeries are made here.
        """

        index_names_set = set(df.index.names)
        # Simple datetime index is always compatible
        if isinstance(df.index, pd.DatetimeIndex):
            return True
        # Check MultiIndex with datetime values
        if isinstance(df.index, pd.MultiIndex):
            logger.info("Check index structure")
            has_datetime = isinstance(
                df.index.get_level_values(0), pd.DatetimeIndex
            ) or isinstance(df.index.get_level_values(1), pd.DatetimeIndex)
            if not isinstance(df.columns, pd.MultiIndex):
                if index_names_set == set(TimeSeries.ROW_INDEX_NAMES) and has_datetime:
                    logger.info("Index is MultiIndex with datetime values.")
                    return True

            # Check full MultiIndex structure
        if isinstance(df.columns, pd.MultiIndex):
            logger.info(
                "Check columns MultiIndex structure. One caveat is that df.index is not DateTimeIndex, casting to datetime is done in align_format()."
            )
            column_names_set = set(df.columns.names)
            are_levels_correct = TimeSeries._check_column_levels(df, representation)
            return (
                index_names_set == set(TimeSeries.ROW_INDEX_NAMES)
                and column_names_set == set(TimeSeries.COL_INDEX_NAMES)
                and are_levels_correct
            )

        return False

    def align_format(self, df: pd.DataFrame):

        if (
            set(TimeSeries.COL_INDEX_NAMES) == set(df.columns.names)
            and TimeSeries.COL_INDEX_NAMES != df.columns.names
        ):
            logger.info("Reordering column names to match expected format.")
            df.columns = df.columns.reorder_levels(
                [df.columns.names.index(name) for name in TimeSeries.COL_INDEX_NAMES]
            )
        if (
            set(TimeSeries.ROW_INDEX_NAMES) == set(df.index.names)
            and TimeSeries.ROW_INDEX_NAMES != df.index.names
        ):
            logger.info("Reordering index names to match expected format.")
            df.index = df.index.reorder_levels(
                [df.index.names.index(name) for name in TimeSeries.ROW_INDEX_NAMES]
            )
        # Casting the index to datetime format if it is a MultiIndex with 'time_stamp'
        if isinstance(df.index, pd.MultiIndex) and "time_stamp" in df.index.names:
            logger.info("Casting the index to datetime format")
            try:
                df.index = pd.MultiIndex.from_arrays(
                    [
                        df.index.get_level_values("offset"),
                        pd.to_datetime(df.index.get_level_values("time_stamp")),
                    ],
                    names=TimeSeries.ROW_INDEX_NAMES,
                )
            except Exception as e:
                logger.error(f"Failed to convert 'time_stamp' to datetime: {e}")
                raise ValueError(
                    f"Cannot convert index level 'time_stamp' to datetime: {e}"
                )
        # Creating multiindex if needed
        if not isinstance(df.index, pd.MultiIndex):
            df.index = pd.MultiIndex.from_product(
                [[pd.Timedelta(0)], df.index], names=TimeSeries.ROW_INDEX_NAMES
            )
        if self.representation == TimeSeries.DETERM_REP:

            if not isinstance(df.columns, pd.MultiIndex):
                df.columns = pd.MultiIndex.from_product(
                    [df.columns, ["value"]], names=TimeSeries.COL_INDEX_NAMES
                )
        elif self.representation == TimeSeries.QUANTILE_REP:
            if self.quantiles is None:
                logger.error("Quantiles must be specified for quantile representation.")
                raise ValueError(
                    "Quantiles must be specified for quantile representation."
                )
            # if not isinstance(df.index, pd.MultiIndex):
            #     df.index = pd.MultiIndex.from_product(
            #         [[pd.Timedelta(0)], df.index], names=TimeSeries.ROW_INDEX_NAMES
            #     )
            if not isinstance(df.columns, pd.MultiIndex):
                df.columns = pd.MultiIndex.from_product(
                    [["target"], self.quantiles], names=TimeSeries.COL_INDEX_NAMES
                )
            else:
                # Rename the outer column levels to needed format
                df.columns.names = TimeSeries.COL_INDEX_NAMES
                # Dynamic relabeling of inner column level to match quantiles
                inner_levels = sorted(set(level[1] for level in df.columns))
                if len(inner_levels) == len(self.quantiles):
                    mapping = dict(zip(inner_levels, self.quantiles))
                    df.columns = pd.MultiIndex.from_tuples(
                        [(outer, mapping[inner]) for outer, inner in df.columns],
                        names=df.columns.names,
                    )
                else:
                    logger.error(
                        "Cannot map inner column levels to quantiles: mismatched length."
                    )
                    raise ValueError(
                        "Cannot map inner column levels to quantiles: mismatched length."
                    )
        elif self.representation == TimeSeries.SAMPLE_REP:
            # raise NotImplementedError("Sample representation not implemented yet.")
            logger.info(
                "Aligning DataFrame for sample representation. At this point it is assumed that all the columns are samples."
            )
            if not isinstance(df.columns, pd.MultiIndex):
                df.columns = pd.MultiIndex.from_product(
                    [["target"], range(len(df.columns))],
                    names=TimeSeries.COL_INDEX_NAMES,
                )
            else:
                # Rename the outer column levels to needed format
                df.columns.names = TimeSeries.COL_INDEX_NAMES
                # Dynamic relabeling of inner column level to match samples
                inner_levels = sorted(set(level[1] for level in df.columns))
                if len(inner_levels) == len(df.columns):
                    mapping = dict(zip(inner_levels, range(len(df.columns))))
                    df.columns = pd.MultiIndex.from_tuples(
                        [(outer, mapping[inner]) for outer, inner in df.columns],
                        names=df.columns.names,
                    )
                else:
                    logger.error(
                        "Cannot map inner column levels to samples: mismatched length."
                    )
                    raise ValueError(
                        "Cannot map inner column levels to samples: mismatched length."
                    )

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

    def shift_to_horizon(self, horizon: int = 1, in_place: bool = False):
        """
        Shifts the time series data forward by a specified horizon.

        This method moves the values in the time series `horizon` steps forward,
        effectively aligning each timestamp with the value that occurs `horizon`
        steps ahead. The index offsets are adjusted accordingly to reflect the shift.

        Parameters
        ----------
        horizon : int, default=1
            The number of steps to shift the time series forward. Positive values
            move the series forward in time.
        in_place : bool, default=False
            If True, modifies the current TimeSeries object and returns `self`.
            If False, returns a new TimeSeries instance with the shifted data.

        Returns
        -------
        TimeSeries
            The shifted TimeSeries. If `in_place=True`, returns the current instance.
            Otherwise, returns a new TimeSeries object with shifted data.

        Notes
        -----
        - The shifting operation fills the vacated positions with NaNs.
        - The MultiIndex of the DataFrame is updated so that the first level (offsets)
        is incremented by `horizon * freq`, while the second level (original timestamps)
        remains unchanged.
        """
        if self.freq is None:
            raise ValueError(
                "A regular frequency must be defined in the TimeSeries instance."
            )
        
        if len(self._offsets) > 1:
            raise ValueError("Shifting is not supported for TimeSeries with offsets other than 0.")
        
        offset_delta = pd.Timedelta(horizon,self.freq)
        shifted_data = self.data.shift(-horizon)
        shifted_index = pd.MultiIndex.from_arrays(
            [
                shifted_data.index.get_level_values(0) + offset_delta,
                shifted_data.index.get_level_values(1),
            ],
            names=TimeSeries.ROW_INDEX_NAMES
        )
        shifted_data.set_index(shifted_index,inplace=True)
        if in_place:
            self.data = shifted_data
            return self
        else:
            return TimeSeries(shifted_data)
        
    def shift_to_repeat_to_multihorizon(horizon: int = 1, in_place: bool = False):
        pass

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

    def by_time(self, horizon: Optional[Union[int, pd.Timedelta]] = None):
        """
        Returns a DataFrame with 'time_stamp' as the outer index.

        Parameters
        ----------
        horizon : int or pd.Timedelta, optional
            - If an `int` is provided, it is interpreted as an offset in hours.
            - If a `pd.Timedelta` is provided, it is used directly to select a specific offset.
            - If None, the MultiIndex is reordered to make 'time_stamp' the outer index.

        Returns
        -------
        pd.DataFrame
            A filtered or reindexed view of the time series data.

        Raises
        ------
        ValueError
            If the horizon is not one of the supported types.
        """
        if horizon is not None:
            if isinstance(horizon, int):
                horizon = self._offsets[horizon]
            elif not isinstance(horizon, pd.Timedelta):
                raise ValueError("Horizon must be an int or pd.Timedelta.")

            # Filter and return only that offset (drops 'offset' level)
            return self.data.xs(horizon, level="offset")

        # Otherwise, reorder index to make time_stamp the outer index
        return self.data.swaplevel("offset", "time_stamp").sort_index(
            level="time_stamp"
        )

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

    def get_feature_slice(self, index: List[str], copy: bool = False) -> TimeSeries:
        """
        Extracts a subset of the data based on the specified columns.
        Representations (level 1 of the column index) are carried over to the new `TimeSeries`.

        Parameters
        ----------
            feature List[str]: The names of the feature to keep in the returned `TimeSeries`
            copy bool, optional: Whether to copy the underlying data. Defaults to False.

        Returns
        -------
        TimeSeries
            A subseries containing a selection by feature.

        Raises
        ------
        TypeError
            If index is not a List of strings
        """
        if (not isinstance(index, List)) or (
            not all([isinstance(i, str) for i in index])
        ):
            raise TypeError("feature must be a list of strings")

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
                    return self.data.index.get_level_values(1)[i]
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
        return len(self.data.index.get_level_values(1))

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
                - type List[str] is interpreted as subset of feature to select for and is forwarded to get_feature
        Returns
        -------
        TimeSeries
            A subseries containing a selection by time or by feature.

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

    def _perform_mixed_operation_core(
        self,
        other: TimeSeries,
        operation_func: Callable[[pd.Series, pd.Series], pd.Series],
        inplace: bool,
    ) -> TimeSeries:
        main_ts = self
        determ_ts = other
        if not (
            main_ts.representation in {TimeSeries.QUANTILE_REP, TimeSeries.SAMPLE_REP}
            and determ_ts.representation == TimeSeries.DETERM_REP
        ):
            # This check is a safeguard; higher-level calls should prevent this state
            raise ValueError(
                "Mixed operation core requires main_ts to be QUANTILE/SAMPLE and other to be DETERM_REP."
            )

        # Decide which DataFrame to operate on: a copy or self.data directly
        target_df = main_ts.data if inplace else main_ts.data.copy()

        quantiles_to_preserve = (
            main_ts.quantiles
            if main_ts.representation == TimeSeries.QUANTILE_REP
            else None
        )

        for feature in target_df.columns.get_level_values(0).unique():

            det_series = determ_ts.data.xs((feature, "value"), axis=1)

            for sub_column in target_df[feature].columns.unique():
                # Perform the operation and assign back directly to target_df
                target_df.loc[:, (feature, sub_column)] = operation_func(
                    target_df.loc[:, (feature, sub_column)], det_series
                )

        if inplace:
            self.data = target_df  # In-place modification, though target_df is already self.data
            return self
        else:
            return TimeSeries(
                data=target_df,
                representation=main_ts.representation,
                quantiles=quantiles_to_preserve,
            )

    def _check_operation_compatibility(self, other: TimeSeries):
        """
        Helper to check compatibility for binary operations like addition or subtraction.
        """
        if not isinstance(other, TimeSeries):
            raise TypeError("Can only operate with another TimeSeries object.")

        if (
            self.data.index.names != other.data.index.names
            or self.data.columns.names != other.data.columns.names
        ):
            raise ValueError(
                "TimeSeries objects must have the same index and column names to perform this operation."
            )
        if not self.data.index.equals(other.data.index):
            raise ValueError(f"TimeSeries indices do not match:\n"
            f"self.index = {self.data.index}\n"
            f"other.index = {other.data.index}")
    def __neg__(self) -> TimeSeries:
        """
        Return a new TimeSeries instance with all values negated.
        The negation is performed according to the current representation of the TimeSeries:
            - For DETERMINISTIC representation, the 'value' column is negated.
            - For QUANTILE representation, all quantile columns for each feature are negated.
            - For SAMPLE representation, all sample columns for each feature are negated.
        Returns:
            TimeSeries: A new TimeSeries object with negated data.
        Raises:
            InvalidRepresentationFormat: If the current representation is not supported.
        """

        negated_data = self.data.copy()
        if self.representation == TimeSeries.DETERM_REP:
            negated_data.loc[
                :, (negated_data.columns.get_level_values(0), "value")
            ] *= -1
        elif self.representation in {TimeSeries.QUANTILE_REP, TimeSeries.SAMPLE_REP}:
            for feature in negated_data.columns.get_level_values(0).unique():
                # Get quantile names from the column itself to be robust
                for sub_column in negated_data[feature].columns.unique():
                    negated_data.loc[:, (feature, sub_column)] *= -1
        else:
            raise InvalidRepresentationFormat(
                "Provided representation is not compatible"
            )
        ts_kwargs = {
            "data": negated_data,
            "representation": self.representation,
        }
        if self.representation == TimeSeries.QUANTILE_REP:
            ts_kwargs["quantiles"] = self.quantiles  # Preserve quantiles list
        return TimeSeries(**ts_kwargs)

    def __add__(self, other: "TimeSeries") -> "TimeSeries":
        """
        Adds two TimeSeries objects together.
        Returns a new TimeSeries object containing the sum of the data.
        """
        self._check_operation_compatibility(other)

        # Handle mixed representations (QUANTILE/SAMPLE + DETERM)
        if (
            self.representation in {TimeSeries.QUANTILE_REP, TimeSeries.SAMPLE_REP}
            and other.representation == TimeSeries.DETERM_REP
        ):
            return self._perform_mixed_operation_core(
                other, operation_func=lambda x, y: x + y, inplace=False
            )
        elif (
            other.representation in {TimeSeries.QUANTILE_REP, TimeSeries.SAMPLE_REP}
            and self.representation == TimeSeries.DETERM_REP
        ):
            return other._perform_mixed_operation_core(
                self, operation_func=lambda x, y: x + y, inplace=False
            )

        # Handle same representations (or other combinations with direct pandas add)
        new_data = self.data.add(other.data, fill_value=0)
        ts_kwargs = {
            "data": new_data,
            "representation": self.representation,
        }
        if self.representation == TimeSeries.QUANTILE_REP:
            ts_kwargs["quantiles"] = self.quantiles
        return TimeSeries(**ts_kwargs)

    def __sub__(self, other: TimeSeries) -> TimeSeries:
        """
        Subtracts another TimeSeries object from this one.

        Parameters
        ----------
        other : TimeSeries
            The TimeSeries to subtract from this one.

        Returns
        -------
        TimeSeries
            A new TimeSeries object containing the difference of the data.
        """
        self._check_operation_compatibility(other)

        return self.__add__(-other)

    def __mul__(self, other: Union[int, float, TimeSeries]) -> TimeSeries:
        """
        Multiplies the TimeSeries data by a scalar value.

        Parameters
        ----------
        scalar : int
            The scalar value to multiply the TimeSeries data by.

        Returns
        -------
        TimeSeries
            A new TimeSeries object with the data multiplied by the scalar.
        """
        if isinstance(other, (int, float)):
            # raise TypeError("Can only multiply by a scalar (int or float).")
            new_data = self.data * other
        elif isinstance(other, TimeSeries):
            if self.data.shape != other.data.shape:
                logger.info("Mul cannot be applied, different shapes")
                # Need to think about multiplication of different representations
                raise ValueError("TimeSeries objects have different shapes")
            new_data = self.data * other.data
        else:
            raise TypeError(
                "Can only multiply by a scalar(int,float) or by other TimeSeries object"
            )
        return TimeSeries(
            data=new_data, representation=self.representation, quantiles=self.quantiles
        )

    def __rmul__(self, scalar: Union[int, float]) -> TimeSeries:
        """
        Implements the right multiplication operation for the TimeSeries object.

        Parameters
        ----------
        scalar : int or float
            The scalar value to multiply the TimeSeries data by.

        Returns
        -------
        TimeSeries
            A new TimeSeries object with the data multiplied by the scalar.

        Raises
        ------
        TypeError
            If the scalar is not an int or float.
        """

        return self.__mul__(scalar)

    def __truediv__(self, scalar: Union[int, float]):
        """
        Divides the TimeSeries data by a scalar value.

        Parameters
        ----------
        scalar : int or float
            The scalar value to divide the TimeSeries data by.

        Returns
        -------
        TimeSeries
            A new TimeSeries object with the data divided by the scalar.

        Raises
        ------
        TypeError
            If the scalar is not an int or float.
        ZeroDivisionError
            If the scalar is zero.
        """
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only divide by scalar (int or float)")
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide TimeSeries by zero.")
        return self.__mul__(1 / scalar)

    def __abs__(self):
        """
        Return a new TimeSeries instance with the absolute values of the data.

        This method computes the element-wise absolute value of the underlying data
        and returns a new TimeSeries object with the same representation and, if applicable,
        the same quantiles.

        Returns:
            TimeSeries: A new TimeSeries object with absolute values of the original data.
        """
        new_data = self.data.abs()
        ts_kwargs = {
            "data": new_data,
            "representation": self.representation,
        }
        if self.representation == TimeSeries.QUANTILE_REP:
            ts_kwargs["quantiles"] = self.quantiles

        return TimeSeries(**ts_kwargs)

    def __iadd__(self, other: TimeSeries) -> Self:
        """
        Performs in-place addition with another TimeSeries object.

        This method modifies the current TimeSeries instance by adding the values from another
        TimeSeries object (`other`). The operation is performed in-place, updating the data and
        representation of `self` as needed. Handles mixed representations (e.g., deterministic with
        quantile/sample) and ensures compatibility before performing the operation.

        Parameters
        ----------
        other : TimeSeries
            The TimeSeries object to add to this instance.

        Returns
        -------
        self : TimeSeries
            The modified TimeSeries instance after in-place addition.

        Raises
        ------
        ValueError
            If the two TimeSeries objects are not compatible for addition.
        """
        self._check_operation_compatibility(other)

        # Handle mixed representations (QUANTILE/SAMPLE + DETERM)
        if (
            self.representation in {TimeSeries.QUANTILE_REP, TimeSeries.SAMPLE_REP}
            and other.representation == TimeSeries.DETERM_REP
        ):
            # If self is quantile/sample and other is deterministic, perform inplace operation
            return self._perform_mixed_operation_core(
                other, operation_func=lambda x, y: x + y, inplace=True
            )
        elif (
            other.representation in {TimeSeries.QUANTILE_REP, TimeSeries.SAMPLE_REP}
            and self.representation == TimeSeries.DETERM_REP
        ):
            temp_result = other._perform_mixed_operation_core(
                self, operation_func=lambda x, y: x + y, inplace=False
            )

            # The caveat here is that if self is deterministic inplace addition to quantiles/sample, will change the data to quantile/sample representation
            logger.info("Attention the TS object's data is not Deterministic now")
            self.data = temp_result.data
            self.representation = temp_result.representation
            self.quantiles = temp_result.quantiles
            return self

        self.data = self.data.add(other.data, fill_value=0)
        return self

    def __isub__(self, other: TimeSeries) -> Self:
        """
        Implements the in-place subtraction operator ( -= ) for TimeSeries objects.

        Parameters
        ----------
        other : TimeSeries
            The TimeSeries instance to subtract from self.

        Returns
        -------
        self
            The updated TimeSeries instance after subtraction.
        """
        return self.__iadd__(-other)

    def __imul__(self, other: Union[int, float, TimeSeries]) -> Self:
        """
        Implements in-place multiplication of the time series data by a scalar.

        Parameters
        ----------
        scalar : int or float
            The scalar value to multiply the time series data by.

        Returns
        -------
        self : Timeseries
            The modified instance with updated data.

        Raises
        ------
        TypeError
            If `scalar` is not an int or float.
        """
        if isinstance(other, (int, float)):
            # raise TypeError("Can only multiply by a scalar (int or float).")
            self.data *= other
        elif isinstance(other, TimeSeries):
            if self.data.shape != other.data.shape:
                logger.info("Mul cannot be applied, different shapes")
                # Need to think about multiplication of different representations
                raise ValueError("TimeSeries objects have different shapes")
            self.data *= other.data
        else:
            raise TypeError(
                "Can only multiply by a scalar(int,float) or by other TimeSeries object"
            )
        return self

    def __itruediv__(self, scalar: Union[int, float]) -> Self:
        """
        In-place division: divides the TimeSeries data by a scalar value, modifying self.

        Parameters
        ----------
        scalar : int or float
            The scalar value to divide the TimeSeries data by.

        Returns
        -------
        self
            The updated TimeSeries object after division.

        Raises
        ------
        TypeError
            If the scalar is not an int or float.
        ZeroDivisionError
            If the scalar is zero.
        """
        if not isinstance(scalar, (int, float)):
            raise TypeError("Can only divide by scalar (int or float)")
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide TimeSeries by zero.")
        return self.__imul__(1 / scalar)
