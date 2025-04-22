from ..modeladapter import ModelAdapter, ModelAdapterError
from ...data.timeseries import TimeSeries
from typing import List,Optional, Any, Union, Tuple
import logging
from pathlib import Path
import pandas as pd
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.dataprocessing.transformers import Scaler
from darts import TimeSeries as DartsTimeSeries

class DartsModelAdapter(ModelAdapter):
    def __init__(self, model:ForecastingModel, 
                 target:dict[int,TimeSeries] = None,
                 known:dict[int,TimeSeries] = None,
                 observed:dict[int,TimeSeries] = None,
                 static:dict[int,TimeSeries] = None,
                 time_col:str = 'time_stamp',
                 group_col:str = 'room_id',
                 freq:str = '60min',
                 *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.model = model
        self.target = target
        self.known = known
        self.observed = observed
        self.static = static
        self.fit_args = None
        self.time_col = time_col
        self.group_col = group_col
        self.freq = freq
        
    def fit(self,**kwargs) -> None:
        super().fit(**kwargs)
        self.fit_args = {'target':self.to_model_format(ts = self.target)}
        if self.model.supports_future_covariates and self.known is not None:
            self.fit_args["future_covariates"] = self.to_model_format(ts = self.known)

        if self.model.supports_past_covariates and self.observed is not None:
            self.fit_args["past_covariates"] = self.to_model_format(ts = self.observed)
            
        if self.model.supports_static_covariates and self.static is not None:
            self.fit_args["static_covariates"] = self.to_model_format(ts = self.static)

    def flatten_ts(self, ts: dict[int, TimeSeries]) -> pd.DataFrame:
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
        #TODO revise the logic of the flatten_timeseries_df function
        for id, ts_obj in ts.items():
            flat_df = flatten_timeseries_df(ts_obj)
            flat_df[self.group_col] = id
            all_dfs.append(flat_df)

        return pd.concat(all_dfs, ignore_index=True)
        
    def to_model_format(self, ts: dict[int,TimeSeries]) -> DartsTimeSeries:
        """
        Converts a dictionary of TimeSeries objects into a DartsTimeSeries object.
        This method takes a dictionary where the keys are integers and the values are 
        TimeSeries objects, flattens the data, and converts it into a DartsTimeSeries 
        object. The conversion can handle grouped data if a grouping column is specified.
        Args:
            ts (dict[int, TimeSeries]): A dictionary where keys are integers and values 
                are TimeSeries objects to be converted.
        Returns:
            DartsTimeSeries: A DartsTimeSeries object created from the input data.
        Notes:
            - If `self.group_col` is specified, the method uses `from_group_dataframe` 
              to create the DartsTimeSeries object, grouping by the specified column.
            - If `self.group_col` is not specified, the method uses `from_dataframe` 
              to create the DartsTimeSeries object.
            - The `value_cols` are determined by excluding the group and time columns 
              from the flattened data.
        """

     
        
        data = self.flatten_ts(ts)
        value_cols = [col for col in data.columns if col not in [self.group_col, self.time_col]]
        data[self.time_col] = pd.to_datetime(data[self.time_col])
        data[self.time_col] = data[self.time_col].dt.tz_localize(None)
        if self.group_col:
            darts_ts = DartsTimeSeries.from_group_dataframe(data,time_col=self.time_col,group_cols=self.group_col, value_cols=value_cols, freq=self.freq)
        else:
            darts_ts = DartsTimeSeries.from_dataframe(data,time_col=self.time_col,value_cols=value_cols, freq=self.freq)
        return darts_ts
    
    def split_series(self,
        ts: Union[DartsTimeSeries, List[DartsTimeSeries]],
        split: Union[float, pd.Timestamp]
    ) -> Union[
        Tuple[DartsTimeSeries, DartsTimeSeries],
        Tuple[List[DartsTimeSeries], List[DartsTimeSeries]]
    ]:
        """
        Splits Darts TimeSeries or list of TimeSeries into training and testing sets.

        Parameters
        ----------
        ts : DartsTimeSeries or List[DartsTimeSeries]
            The time series or list of time series to split.

        split : float or pd.Timestamp
            - If float (0 < split < 1), splits by percentage of the series length.
            - If pd.Timestamp, splits at that specific timestamp.

        Returns
        -------
        Tuple
            - Single TimeSeries: (train_ts, test_ts)
            - List of TimeSeries: (list of train_ts, list of test_ts)
        """

        def _split_single(series: DartsTimeSeries):
            if isinstance(split, float):
                if not 0 < split < 1:
                    raise ValueError("Split ratio must be between 0 and 1.")
                split_index = int(len(series) * split)
                return series[:split_index], series[split_index:]
            elif isinstance(split, pd.Timestamp):
                if split not in series.time_index:
                    raise ValueError(f"Timestamp {split} not in series index.")
                return series.split_before(split)
            else:
                raise TypeError("Split must be float or pd.Timestamp.")

        if isinstance(ts, list):
            return zip(*[_split_single(s) for s in ts])
        else:
            return _split_single(ts)
        
    def to_time_series(ts:DartsTimeSeries):
        #Need to think of the way to implement this method
        return super().to_time_series()
    
    def predict(self):
        raise NotImplementedError("Subclasses must implement this method.")
    def tune(self):
        raise NotImplementedError("Subclasses must implement this method.")
    def load(self):
        raise NotImplementedError("Subclasses must implement this method.")
    def save(self):
        raise NotImplementedError("Subclasses must implement this method.")
    