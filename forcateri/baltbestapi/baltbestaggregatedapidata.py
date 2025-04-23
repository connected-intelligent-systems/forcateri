from typing import List,Optional, Any, Union, Tuple
from forcateri.baltbestapi.baltbestapidata import BaltBestAPIData
from ..data.timeseries import TimeSeries
import pandas as pd
from datetime import datetime
import logging

class BaltBestAggregatedAPIData(BaltBestAPIData):
    
    def __init__(self,**kwargs):
        super().__init__(name=kwargs['name'], url=kwargs['url'],local_copy = kwargs['local_copy'])
        self.ts: dict[int,TimeSeries] = None
        self.target: dict[int,TimeSeries]  = None
        self.known: dict[int,TimeSeries]  = None
        self.observed: dict[int,TimeSeries]  = None
        self.static: dict[int,TimeSeries]  = None

    def get_data(self):
        super().get_data()

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
        self.ts, self.ts_dict = TimeSeries.from_group_df(df=df,
                                               group_col = self.group_col,
                                               time_col=self.time_col,
                                               value_cols=self.value_cols,
                                               freq = self.freq
                                               )

    def _separate_ts(self, target: str, known: Union[str, list[str]], observed: Union[str, list[str]], static: Union[str, list[str]] = None):
        """
        Separate the TimeSeries instance into target, known, observed, and static components.

        The method processes the `ts` attribute, which is expected to be a TimeSeries object,
        and assigns its components to the instance's attributes: `target`, `known`, `observed`, and `static`.

        Parameters
        ----------
        target : str
            The column name for the target variable.
        known : Union[str, list[str]]
            The column name(s) for the known variables.
        observed : Union[str, list[str]]
            The column name(s) for the observed variables.
        static : Union[str, list[str]], optional
            The column name(s) for the static variables, by default None.

        Returns
        -------
        None
            This method modifies the instance's attributes to store the separated components of the TimeSeries.
        """
        
        if self.ts is None:
            raise ValueError("Fetch the data first")
        self.target = []
        self.known = []
        self.observed = []
        self.static = []
        for ts_obj in self.ts:
            data = ts_obj.data
            if target is not None:
                self.target.append(data[[target]])
            if known is not None:
                self.known.append(data[[known]] if isinstance(known, str) else data[known])
            if observed is not None:
                self.observed.append(data[[observed]] if isinstance(observed, str) else data[observed])
            if static is not None:
                self.static.append(data[[static]] if isinstance(static, str) else data[static])
        

    def is_up2date(self):
        #TODO update the logic later
        self.last_updated = datetime.now()
        return True
    
    def update_local_copy(self):
        #TODO update the logic later
        pass
        
    def _fetch_data_from_api(self):
        raise NotImplementedError("Subclasses must implement this method.")

