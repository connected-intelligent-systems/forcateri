from .datasource import DataSource
from typing import List, Tuple, Union, Dict
from .timeseries import TimeSeries
import pandas as pd
from datetime import datetime
from enum import Enum

Cutoff = Tuple[
    Union[int, float, str, datetime, pd.Timestamp],
    Union[int, float, str, datetime, pd.Timestamp],
]

class SeriesRole(Enum):
    TARGET = "target"
    OBSERVED = "observed"
    KNOWN = "known"
    STATIC = "static"

class DataProvider:
    """
    A class that manages data sources and compiles their data into train, validation and test sets.
    """

    def __init__(
        self,
        data_sources: List[DataSource],
        roles: Dict[str, SeriesRole],
        splits: Union[Cutoff, List[Cutoff]] = (1.0 / 3.0, 2.0 / 3.0),
    ):
        """
        Initializes the DataProvider class.
        """
        self.roles = roles
        self.splits = splits
        self.data_sources = data_sources
        self.target=[]
        self.known=[]
        self.observed=[]
        self.static=[]
        self.roles_reversed = dict(zip(roles.values(), roles.keys()))
        self._separate_ts()

    def _separate_ts(self):
        """
        Separates the time series into target, known, observed and static based on roles.
        """
        for data_source in self.data_sources:
            data_list = data_source.get_data()
            for ts_obj in data_list:
                
                self.target.append(ts_obj.slice(columns = self.roles_reversed[SeriesRole.TARGET]))
                self.known.append(ts_obj.slice(columns = self.roles_reversed[SeriesRole.KNOWN]))
                self.observed.append(ts_obj.slice(columns = self.roles_reversed[SeriesRole.OBSERVED]))


    #KNWON, OBSERVED, TARGET            


    def _get_split_set(self, split_type: str):
      """
      Generic method to retrieve a dataset (train, validation, or test) based on splits.

      Args:
        split_type (str): The type of split to retrieve. Can be 'train', 'val', or 'test'.

      Returns:
        list: A list of tuples, where each tuple contains:
          - target: A target time series for the specified split.
          - known: A known time series for the specified split.
          - observed: An observed time series for the specified split.
          - static: The static data, unchanged across splits.
      """
      start, end = self.splits

      if split_type == "train":

        def condition(ts):
          if isinstance(start, (pd.Timestamp, datetime)):
            return ts.index.get_level_values('time_stamp') < start
          elif isinstance(start, int):
            return slice(None, start)
          else:
            return slice(None, int(len(ts) * start))
          
      elif split_type == "val":

        def condition(ts):
          if isinstance(start, (pd.Timestamp, datetime)):
            return (ts.index.get_level_values('time_stamp') >= start) & (ts.index.get_level_values('time_stamp') < end)
          elif isinstance(start, int) and isinstance(end, int):
            return slice(start, end)
          else:
            return slice(int(len(ts) * start), int(len(ts) * end))
      elif split_type == "test":

        def condition(ts):
          if isinstance(end, (pd.Timestamp, datetime)):
            return ts.index.get_level_values('time_stamp') >= end
          elif isinstance(end, int):
            return slice(end, None)
          else:
            return slice(int(len(ts) * end), None)
      else:
        raise ValueError("Invalid split_type. Must be 'train', 'val', or 'test'.")

      result = []
      for target_ts, known_ts, observed_ts in zip(self.target, self.known, self.observed):
        target_split = target_ts.loc[condition(target_ts)] if isinstance(condition(target_ts), pd.Series) else target_ts[condition(target_ts)]
        known_split = known_ts.loc[condition(known_ts)] if isinstance(condition(known_ts), pd.Series) else known_ts[condition(known_ts)]
        observed_split = observed_ts.loc[condition(observed_ts)] if isinstance(condition(observed_ts), pd.Series) else observed_ts[condition(observed_ts)]
        result.append((target_split, known_split, observed_split))

      return result

    def get_train_set(self):
      return self._get_split_set("train")

    def get_val_set(self):
      return self._get_split_set("val")

    def get_test_set(self):
      return self._get_split_set("test")
    

