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
        """
        start, end = self.splits
        val_target = [ts[int(len(ts) * start):int(len(ts) * end)] for ts in self.target]
        val_known = [ts[int(len(ts) * start):int(len(ts) * end)] for ts in self.known]
        val_observed = [ts[int(len(ts) * start):int(len(ts) * end)] for ts in self.observed]
        val_static = self.static  # Static data remains the same across splits

        return {
            "target": val_target,
            "known": val_known,
            "observed": val_observed,
            "static": val_static,
        }
    def get_train_set(self):
        """
        Returns train set based on the splits parameter.
        """
    
        start, _ = self.splits
        train_target = [ts[:int(len(ts) * start)] for ts in self.target]
        train_known = [ts[:int(len(ts) * start)] for ts in self.known]
        train_observed = [ts[:int(len(ts) * start)] for ts in self.observed]
        train_static = self.static  # Static data remains the same across splits

        return {
            "target": train_target,
            "known": train_known,
            "observed": train_observed,
            "static": train_static,
        }
    def get_test_set(self):
        """
        Returns test set based on the splits parameter.
        """
        _, end = self.splits
        test_target = [ts[int(len(ts) * end):] for ts in self.target]
        test_known = [ts[int(len(ts) * end):] for ts in self.known]
        test_observed = [ts[int(len(ts) * end):] for ts in self.observed]
        test_static = self.static  # Static data remains the same across splits

        return {
            "target": test_target,
            "known": test_known,
            "observed": test_observed,
            "static": test_static,
        }
    

    #Need to split the series into train, val and test sets. Such that all model adapter are trained with the same scenario.
    #The logic of splitting inside DataProvider.
    #Division into target, known, observed here as well.