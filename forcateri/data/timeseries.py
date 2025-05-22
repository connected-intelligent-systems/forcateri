from __future__ import annotations

from datetime import datetime
import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TimeSeries:
    
    DETERM_REP = "determ"
    QUANTILE_REP = "quantile"
    SAMPLE_REP = "sample"
    def __init__(
        self,
        data: pd.DataFrame,
        representation=None,
        quantiles: Optional[List[float]] = None,
    ):
        if representation is None:
            representation = TimeSeries.DETERM_REP
        self.representation = representation
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame")

        # If already in internal format (e.g. MultiIndex on both axes), just store it
        if TimeSeries.is_matching_format(data):
            self.data = data.copy()
            logger.info("TimeSeries initialized from internal-format DataFrame.")
        elif TimeSeries.is_compatible_format(data):
            # If the DataFrame is compatible but not in the expected format, align it
            self.data = data.copy()
            self.align_format(self.data)
            logger.info("TimeSeries initialized from compatible-format DataFrame.")
        else:
            logger.info("Raw DataFrame provided")
            raise ValueError("Cannot build ts from the DataFrame")


    @staticmethod
    def is_matching_format(df: pd.DataFrame) -> bool:
        '''
        Checks the structure of the row and the column index and returns true if a data frame
        has the expected format to serve as a TimeSeries data representation.
        No changes to the data or TimeSeries are made here.
        '''
        if not (isinstance(df.index, pd.MultiIndex) and isinstance(
            df.columns, pd.MultiIndex
        )):
            return False
        expected_index_names = ['offset', 'time_stamp']
        expected_column_names = ['feature', 'representation']

        return df.index.names == expected_index_names and df.columns.names == expected_column_names

    @staticmethod
    def is_compatible_format(df:pd.DataFrame) -> bool:
        '''
        Checks the structure of the row and the column index and returns true if all the missing 
        and or mislabeled information can be inferred.
        No changes to the data or TimeSeries are made here.
        '''
        if isinstance(df.index,pd.DatetimeIndex):
            return True
        expected_index_names = {'offset', 'time_stamp'}
        expected_column_names = {'feature', 'representation'}
        index_names_set = set(df.index.names)
        if isinstance(df.columns,pd.MultiIndex):
            column_names_set = set(df.columns.names)
            # Check that all expected names are present (order doesn't matter)
            if index_names_set == expected_index_names and column_names_set == expected_column_names:
                return True
        
        return False
    
    
    def align_format(self,df:pd.DataFrame):
        
        expected_index_names = ['offset', 'time_stamp']
        expected_column_names = ['feature', 'representation']
        if set(expected_column_names) == set(df.columns.names) and expected_column_names != df.columns.names:
            df.columns = df.columns.reorder_levels([df.columns.names.index(name) for name in expected_column_names])
        if set(expected_index_names) == set(df.index.names) and expected_index_names != df.index.names:
            df.index = df.index.reorder_levels([df.index.names.index(name) for name in expected_index_names])
        if self.representation  == 'value':
            if not isinstance(df.index,pd.MultiIndex):
                df.index = pd.MultiIndex.from_product([[pd.Timedelta(0)],df.index],names=expected_index_names)
            if not isinstance(df.columns,pd.MultiIndex):
                df.columns = pd.MultiIndex.from_product([df.columns,["value"]],names=expected_column_names)
        elif self.representation == 'quantile':
            if not isinstance(df.index,pd.MultiIndex):
                df.index = pd.MultiIndex.from_product([[pd.Timedelta(0)],df.index],names=expected_index_names)
            if not isinstance(df.columns,pd.MultiIndex): 
                df.columns = pd.MultiIndex.from_product([df.columns,QUANTILES],names=expected_column_names)
            else:
                #Rename the outer column levels to needed format
                df.columns.names = expected_column_names
                # Dynamic relabeling of inner column level to match quantiles
                inner_levels = sorted(set(level[1] for level in df.columns))
                if len(inner_levels) == len(QUANTILES):
                    mapping = dict(zip(inner_levels, QUANTILES))
                    df.columns = pd.MultiIndex.from_tuples(
                        [(outer, mapping[inner]) for outer, inner in df.columns],
                        names=df.columns.names
                    )
                else:
                    raise ValueError("Cannot map inner column levels to quantiles: mismatched length.")
        elif self.representation == 'sample':
            raise NotImplementedError("Sample representation not implemented yet.")

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
