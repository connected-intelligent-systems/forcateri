from .datasource import DataSource
from typing import List
from .timeseries import TimeSeries


class DataProvider:
    """
    A class that manages data sources and compiles their data into train, validation and test sets.
    """

    def __init__(self, data_sources: List[DataSource]):
        """
        Initializes the DataProvider class.
        """
        self.data_sources = data_sources
        self.target=[]
        self.known=[]
        self.observed=[]
        self.static=[]
        self._separate_ts()

    def _separate_ts(self):
        """
        Separates the time series into target, known, observed and static.
        """
        for data_source in self.data_sources:
            data_source.get_data()
            for ts_obj in data_source.ts:
                data = ts_obj.data
                if data_source.target is not None:
                    self.target.append(data[[data_source.target]])
                if data_source.known is not None:
                    self.known.append(data[[data_source.known]] if isinstance(data_source.known, str) else data[data_source.known])
                if data_source.observed is not None:
                    self.observed.append(data[[data_source.observed]] if isinstance(data_source.observed, str) else data[data_source.observed])
                if data_source.static is not None:
                    self.static.append(data[[data_source.static]] if isinstance(data_source.static, str) else data[data_source.static])
            


    def get_val_set(self):
        """
        Returns val set.
        """
        pass
    def get_train_set(self):
        """
        Returns train set.
        """
        pass
    def get_test_set(self):
        """
        Returns test set.
        """ 
        pass
    

    #Need to split the series into train, val and test sets. Such that all model adapter are trained with the same scenario.
    #The logic of splitting inside DataProvider.
    #Division into target, known, observed here as well.