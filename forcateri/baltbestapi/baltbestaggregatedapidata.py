from typing import List, Optional
from forcateri.baltbestapi.baltbestapidata import BaltBestAPIData
from ..data.timeseries import TimeSeries
import pandas as pd
from datetime import datetime

class BaltBestAggregatedAPIData(BaltBestAPIData):
    
    def __init__(self,**kwargs):
        super().__init__(name=kwargs['name'], url=kwargs['url'],local_copy = kwargs['local_copy'])
        self.ts: Optional[List[TimeSeries]] = None

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
        self.ts = TimeSeries.from_group_df(df=df,
                                               group_col = self.group_col,
                                               time_col=self.time_col,
                                               value_cols=self.value_cols,
                                               freq = self.freq
                                               )
    def to_pandas(self):
        """
        Converts all time series in the `self.ts` dictionary to a single flattened pandas DataFrame.

        Each time series DataFrame is:
        - Reset to move the timestamp index into a column
        - Stripped of any 'offset' column (if present)
        - Flattened if it contains MultiIndex columns (e.g., ('feature', 'value'))
        - Augmented with a column named `self.group_col` to identify the original series (e.g., room ID)

        Returns
        -------
        pd.DataFrame
            A concatenated DataFrame containing all time series data, with a `time_stamp` column and
            a group column (e.g., 'room_id') indicating the source of each row.
        """
        all_dfs = []

        def flatten_timeseries_df(df: pd.DataFrame) -> pd.DataFrame:
            # Reset index to make 'time_stamp' a column
            df_reset = df.reset_index()

            # Drop the 'offset' column if it's not needed
            if 'offset' in df_reset.columns:
                df_reset = df_reset.drop(columns='offset')

            # Flatten the column MultiIndex
            df_reset.columns = [
                col if not isinstance(col, tuple) else col[0] for col in df_reset.columns
            ]

            # Ensure 'time_stamp' is the first column
            cols = df_reset.columns.tolist()
            if 'time_stamp' in cols:
                cols.insert(0, cols.pop(cols.index('time_stamp')))
                df_reset = df_reset[cols]

            

            return df_reset

        for id, ts_obj in self.ts.items():
            flat_df = flatten_timeseries_df(ts_obj.data)
            flat_df[self.group_col] = id
            all_dfs.append(flat_df)

        return pd.concat(all_dfs, ignore_index=True)

    
    def is_up2date(self):
        #TODO update the logic later
        self.last_updated = datetime.now()
        return True
    
    def update_local_copy(self):
        #TODO update the logic later
        pass
        
    def _fetch_data_from_api(self):
        raise NotImplementedError("Subclasses must implement this method.")

