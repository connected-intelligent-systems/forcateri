import pandas as pd
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
        freq: Optional[Union[str, int]] = 'h'
    ) -> Self:
        """
        Build a deterministic TimeSeries instance from a selection of columns of a DataFrame.
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
            determ_cols = ["value"]
            features = value_cols
            col_dim_names = ["feature", "representation"]
            row_dim_names =["offset", "time_stamp"]
            t0_index = pd.date_range(start=df[time_col].min(), end=df[time_col].max(), freq=freq)
            point_0_index = [pd.Timedelta(0)]
            point_0_row_index = pd.MultiIndex.from_product([point_0_index, t0_index], names=row_dim_names)
            determ_col_index = pd.MultiIndex.from_product([features, determ_cols], names=col_dim_names)
            df = df[features]
            determ_ts = pd.DataFrame(df.values,index=point_0_row_index,columns=determ_col_index)
        else:
            logger.error("Initialization failed: time_col is not provided.")
            raise ValueError(f"Invalid type of time_col: it needs to be of type str.") 
            
            
        return cls(determ_ts)

    def to_samples(self,n_samples:int):
        pass 
    def to_quantiles(self):
        pass 
    def by_time(self,horizon):
        pass 
    def by_horizon(self,t0):
        pass