from typing import List,Optional, Any, Union, Tuple
from forcateri.baltbestapi.baltbestapidata import BaltBestAPIData
from ..data.timeseries import TimeSeries
import pandas as pd
from datetime import datetime
import logging

class BaltBestAggregatedAPIData(BaltBestAPIData):
    
    def __init__(self,**kwargs):
        super().__init__(name=kwargs['name'], url=kwargs['url'],local_copy = kwargs['local_copy'])
        self.ts = []
        self.target:str = kwargs.get('target', None)
        self.group_col:str = kwargs.get('group_col', None)
        self.time_col:str = kwargs.get('time_col', None)
        #self.value_cols:List[str] = kwargs.get('value_cols', None)
        self.freq:str = kwargs.get('freq', '60min')
        self.known:Union[str,List[str]] = kwargs.get('known', None)
        self.observed:Union[str,List[str]] = kwargs.get('observed', None)
        self.static:Union[str,List[str]] = kwargs.get('static', None)
        self.value_cols:List[str] = self._get_value_cols(self.target, self.known, self.observed, self.static)
        self.ts_dict = {}

    def get_data(self):
        super().get_data()
        return self.ts
    def _fetch_from_cache(self):
        """
        Fetch data from a local CSV file, process it by resampling and grouping, and store it as a TimeSeries instance.

        The method performs the following operations:
        - Reads the CSV file into a DataFrame.
        - Converts the time column to datetime format.
        - Groups the data by a specified column and resamples it into 60-minute intervals.
        - Drops the grouping column and resets the index.
        - Converts the resulting DataFrame into a TimeSeries instance.

        Parameters
        ----------
        None

        Returns
        -------
        None
            This method modifies the instance's `ts` attribute to store the resulting TimeSeries object.

        Raises
        ------
        FileNotFoundError
            If the specified local CSV file does not exist.
        ValueError
            If the time column is not present in the DataFrame.
        """
        
        df = pd.read_csv(self.local_copy)
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df = (
            df.set_index(self.time_col)
            .groupby(self.group_col)
            .resample('60min')
            .asfreq()
            .drop(columns=[self.group_col])
            .reset_index())
        self.ts, self.ts_dict = self._from_group_df(df=df,
                                               group_col = self.group_col,
                                               time_col=self.time_col,
                                               value_cols=self.value_cols,
                                               freq = self.freq
                                               )

    def _get_value_cols(self, *args: Union[str, List[str], None]) -> List[str]:
        """
        Safely merges input variables into a single flat list.
        """
        result = []
        for arg in args:
            if arg is None:
                continue
            if isinstance(arg, list):
                result.extend(arg)
            else:
                result.append(arg)
        return result

    def _from_group_df(self, df: pd.DataFrame,
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
            #logger.error("Initialization failed: group_col not found in the DataFrame.")
            raise ValueError(f"Column {group_col} not found in the DataFrame.")
        # if value_cols is None:
        #         value_cols = df.columns[df.columns != time_col]
        unique_group = df[group_col].unique()
        ts_dict = {}
        ts_list = []
        for i, group_id in enumerate(unique_group):
            df_group = df[df[group_col] == group_id]
            ts_instance = TimeSeries.from_dataframe(
                df_group, time_col, value_cols, freq, ts_type
            )
            #ts_dict[group_id] = ts_instance
            ts_list.append(ts_instance)
            ts_dict[i] = group_id
        return ts_list, ts_dict
        

    def is_up2date(self):
        #TODO update the logic later
        self.last_updated = datetime.now()
        return True
    
    def update_local_copy(self):
        #TODO update the logic later
        pass
        
    def _fetch_data_from_api(self):
        raise NotImplementedError("Subclasses must implement this method.")

