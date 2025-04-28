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
        self._separate_ts()

    def _separate_ts(self):
        """
        Separates the time series into target, known, observed and static based on roles.
        """
        for data_source in self.data_sources:
            data_list = data_source.get_data()
            for data in data_list:
                for role, columns in self.roles.items():
                    columns = [columns] if isinstance(columns, str) else columns  
                    if role == SeriesRole.TARGET:
                        self.target.append(data[columns])
                    elif role == SeriesRole.KNOWN:
                        self.known.append(data[columns])
                    elif role == SeriesRole.OBSERVED:
                        self.observed.append(data[columns])
                    elif role == SeriesRole.STATIC:
                        self.static.append(data[columns])
                    else:
                        raise ValueError(f"Unknown role: {role}")
                   


    def get_val_set(self):
        """
        Returns validation set based on the splits parameter.
        Handles both index-based and timestamp-based splits, including MultiIndex.
        """
        start, end = self.splits

        if isinstance(start, (pd.Timestamp, datetime)) and isinstance(end, (pd.Timestamp, datetime)):
            val_target = [ts.loc[(ts.index.get_level_values('time_stamp') >= start) & 
                                 (ts.index.get_level_values('time_stamp') < end)] for ts in self.target]
            val_known = [ts.loc[(ts.index.get_level_values('time_stamp') >= start) & 
                                (ts.index.get_level_values('time_stamp') < end)] for ts in self.known]
            val_observed = [ts.loc[(ts.index.get_level_values('time_stamp') >= start) & 
                                   (ts.index.get_level_values('time_stamp') < end)] for ts in self.observed]
        else:
            val_target = [ts[int(len(ts) * start):int(len(ts) * end)] for ts in self.target]
            val_known = [ts[int(len(ts) * start):int(len(ts) * end)] for ts in self.known]
            val_observed = [ts[int(len(ts) * start):int(len(ts) * end)] for ts in self.observed]

        val_static = self.static  # Static data remains the same across splits

        return (val_target, val_known, val_observed, val_static)
    def get_train_set(self):
        """
        Returns train set based on the splits parameter.
        Handles both index-based and timestamp-based splits, including MultiIndex.
        """
        start, _ = self.splits

        if isinstance(start, (pd.Timestamp, datetime)):
            train_target = [ts.loc[ts.index.get_level_values('time_stamp') < start] for ts in self.target]
            train_known = [ts.loc[ts.index.get_level_values('time_stamp') < start] for ts in self.known]
            train_observed = [ts.loc[ts.index.get_level_values('time_stamp') < start] for ts in self.observed]
        else:
            train_target = [ts[:int(len(ts) * start)] for ts in self.target]
            train_known = [ts[:int(len(ts) * start)] for ts in self.known]
            train_observed = [ts[:int(len(ts) * start)] for ts in self.observed]

        train_static = self.static  # Static data remains the same across splits

        return (train_target, train_known, train_observed, train_static)
    def get_test_set(self):
        """
        Returns test set based on the splits parameter.
        Handles both index-based and timestamp-based splits, including MultiIndex.
        """
        _, end = self.splits

        if isinstance(end, (pd.Timestamp, datetime)):
            test_target = [ts.loc[ts.index.get_level_values('time_stamp') >= end] for ts in self.target]
            test_known = [ts.loc[ts.index.get_level_values('time_stamp') >= end] for ts in self.known]
            test_observed = [ts.loc[ts.index.get_level_values('time_stamp') >= end] for ts in self.observed]
        else:
            test_target = [ts[int(len(ts) * end):] for ts in self.target]
            test_known = [ts[int(len(ts) * end):] for ts in self.known]
            test_observed = [ts[int(len(ts) * end):] for ts in self.observed]

        test_static = self.static  # Static data remains the same across splits

        return (test_target, test_known, test_observed, test_static)
    

