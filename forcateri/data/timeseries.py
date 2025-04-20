import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Self, Sequence, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class TimeSeries:
    def __init__(self, data:pd.DataFrame):
        if not isinstance(data,pd.DataFrame):
            logger.error("Initialization failed: data is not a pandas DataFrame.")
            raise TypeError("Expected a pandas DataFrame instance")
        self.data = data
        logger.info("TimeSeries instance created successfully.")

    @classmethod
    def from_dataframe(                                                 
        cls,
        df:pd.DataFrame, 
        time_col:Optional[str] = None, 
        value_cols: Optional[Union[List[str], str]] = None,
        freq: Optional[Union[str, int]] = 'h',
        ts_type:Optional[str] ='determ'
    ) -> Self:
        """
        Build a TimeSeries instance based on time series type from a selection of columns of a DataFrame.
        One column (or the DataFrame index) has to represent the time,
        and a list of columns `value_cols` has to represent the values for this time series.

        Parameters
        ----------
        df 
          The DataFrame from which to initialize the instance.
        time_col : Optional[str], default None
            The name of the column in the DataFrame that contains time information.
            If provided, this column must exist in the DataFrame.
        value_cols : Optional[Union[List[str], str]], default None
            The name(s) of the column(s) in the DataFrame that contain the values.
            Can be a single column name or a list of column names.
        freq : Optional[Union[str, int]], default None
            The frequency of the time series data.
        ts_type : Optional[str], default 'determ'
             The type of the time series, use 'quantile' - for quantile forecasts, 'determ' - for deterministic series and 'sampled' - for sampled series

        Returns
        -------
        TimeSeries
            A univariate or multivariate deterministic TimeSeries constructed from the inputs.

        Raises
        ------
        ValueError
            If `time_col` is provided but not found in the DataFrame.
        """
        if time_col:
            if time_col not in df.columns:
                logger.error("Initialization failed: time_col not found in the DataFrame.")
                raise ValueError(f"Column {time_col} not found in the DataFrame.")
            
            t0_index = pd.date_range(start=df[time_col].min(), end=df[time_col].max(), freq=freq)
            features = value_cols
            col_dim_names = ["feature", "representation"]
            row_dim_names =["offset", "time_stamp"]
            if ts_type == 'determ':
                determ_cols = ["value"]
                point_0_index = [pd.Timedelta(0)]
                point_0_row_index = pd.MultiIndex.from_product([point_0_index, t0_index], names=row_dim_names)
                determ_col_index = pd.MultiIndex.from_product([features, determ_cols], names=col_dim_names)
                df = df[features]
                determ_ts = pd.DataFrame(df.values,index=point_0_row_index,columns=determ_col_index)
                return cls(determ_ts)
            elif ts_type == 'sampled':
                sampled_cols = [f"s_{i}" for i in range(16)]
                point_1_index = [pd.Timedelta(1, unit="h")] 
                point_1_row_index = pd.MultiIndex.from_product([point_1_index, t0_index], names=row_dim_names)
                sampled_col_index = pd.MultiIndex.from_product([features, sampled_cols], names=col_dim_names)
                sampled_ts = pd.DataFrame(df.loc[:,df.columns!=time_col].values,index=point_1_row_index, columns=sampled_col_index)
                return cls(sampled_ts)
            elif ts_type == 'quantile':
                #The functionality to be checked
                quant_cols = ["q_0.1", "q_0.5", "q_0.9"]
                range_index = pd.to_timedelta(np.arange(1, 25), unit="h")
                range_row_index = pd.MultiIndex.from_product([range_index, t0_index], names=row_dim_names)
                quant_col_index = pd.MultiIndex.from_product([features, quant_cols], names=col_dim_names)
                quant_ts = pd.DataFrame(df.loc[:,df.columns!=time_col].values,index=range_row_index,columns=quant_col_index)
                return cls(quant_ts)
            else:
                logger.error("incorrect ts_type was provided")
                raise ValueError("Invalid type of timeseries was provided")
                
        else:
            logger.error("Initialization failed: time_col is not provided.")
            raise ValueError(f"Invalid type of time_col: it needs to be of type str.") 
            
            
        return None
    
    @classmethod
    def from_group_df(cls,
        df:pd.DataFrame, 
        group_col:str,
        time_col:Optional[str] = None, 
        value_cols: Optional[Union[List[str], str]] = None,
        freq: Optional[Union[str, int]] = 'h',
        ts_type:Optional[str] ='determ',    
    ) -> List[pd.DataFrame]:
        """
        Build TimeSeries instances for each group in the DataFrame.

        This method groups the DataFrame by the specified `group_col` and applies the logic
        from `from_dataframe` to each group. Each group is expected to contain a time column (or index)
        and one or more value columns representing the time series data.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame from which to initialize the instances.
        group_col : str
            The column name used for grouping the data. Each unique value in this column will
            result in a separate TimeSeries instance.
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
            A list of TimeSeries instances, one for each unique group in the DataFrame.

        Raises
        ------
        ValueError
            If `group_col` is not found in the DataFrame.
        """
        if group_col not in df.columns:
            logger.error("Initialization failed: group_col not found in the DataFrame.")
            raise ValueError(f"Column {group_col} not found in the DataFrame.")
        unique_group = df[group_col].unique()
        ts_dict = {}
        for group_id in unique_group:
            df_group = df[df[group_col] == group_id]
            ts_instance  = cls.from_dataframe(df_group,time_col, value_cols,freq,ts_type)
            ts_dict[group_id] = ts_instance
        return ts_dict
    
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
        #TODO from quantiles to samples, This method is not really applicable
        pass

    def to_quantiles(self, quantiles:List[float] = [0.1,0.5,0.9]) -> pd.DataFrame:
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
        pass
    def by_time(self,horizon:Optional[Union[int,pd.Timestamp]]=None):
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
        if isinstance(horizon,pd.Timestamp):
            return self.data.swaplevel(axis=0).loc(horizon) 
        elif isinstance(horizon,int):
            #TODO the logic to handle int horizon
            pass 
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
            forecasts['time_stamp'] = forecasts.index + t0
            forecasts.set_index("time_stamp", inplace=True, drop=True)
            return forecasts
        except KeyError:
            logger.error(f"{t0} not found in forecast data.")
            raise ValueError(f"{t0} offset is not found in the forecast data")
