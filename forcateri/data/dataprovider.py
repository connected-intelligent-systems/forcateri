from .datasource import DataSource
from typing import List, Tuple, Union, Dict, NamedTuple
from .timeseries import TimeSeries
import pandas as pd
from datetime import datetime
from .seriesrole import SeriesRole
from .adapterinput import AdapterInput


Cutoff = Tuple[
    Union[int, float, str, datetime, pd.Timestamp],
    Union[int, float, str, datetime, pd.Timestamp],
]




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

        Args:
            data_sources (List[DataSource]): A list of data sources to be used by the DataProvider.
            roles (Dict[str, SeriesRole]): A dictionary mapping series names to their respective roles.
            splits (Union[Cutoff, List[Cutoff]], optional): The split points for dividing the data. 
                Defaults to (1.0 / 3.0, 2.0 / 3.0).

        Attributes:
            roles (Dict[str, SeriesRole]): Stores the roles of the series.
            splits (Union[Cutoff, List[Cutoff]]): Stores the split points for data division.
            data_sources (List[DataSource]): Stores the provided data sources.
            target (list): A list to store target series.
            known (list): A list to store known series.
            observed (list): A list to store observed series.
            static (Dict[str, float]): A dictionary to store static features.
        """
        self.roles = roles
        self.splits = splits
        self.data_sources = data_sources
        self.target=[]
        self.known=[]
        self.observed=[]
        self.static=Dict[str, float]
        #self.roles_reversed = dict(zip(roles.values(), roles.keys()))
        self._separate_ts()

    def _separate_ts(self):
        """""
        Separates the time series data into target, known, and observed categories 
        based on their assigned roles, and appends the corresponding slices to 
        their respective lists.

        This method uses the `roles` attribute to determine the role of each column 
        in the time series data (e.g., TARGET, KNOWN, OBSERVED). It then iterates 
        through the data sources, retrieves the data, and extracts the relevant 
        feature slices for each role.

        Attributes:
            roles (dict): A dictionary mapping column names to their respective roles 
                          (e.g., SeriesRole.TARGET, SeriesRole.KNOWN, SeriesRole.OBSERVED).
            data_sources (list): A list of data sources containing time series data.
            target (list): A list to store the extracted target feature slices.
            known (list): A list to store the extracted known feature slices.
            observed (list): A list to store the extracted observed feature slices.

        Raises:
            AttributeError: If `roles` or `data_sources` is not properly defined.
        """

        columns_observed = [col for col, role in self.roles.items() if role == SeriesRole.OBSERVED]
        columns_known = [col for col, role in self.roles.items() if role == SeriesRole.KNOWN]   
        columns_target = [col for col, role in self.roles.items() if role == SeriesRole.TARGET]
        for data_source in self.data_sources:
            data_list = data_source.get_data()
            for ts_obj in data_list:
                self.target.append(ts_obj.get_feature_slice(index = columns_target))
                self.known.append(ts_obj.get_feature_slice(index = columns_known))
                self.observed.append(ts_obj.get_feature_slice(index = columns_observed))


    def _get_split_set(self, split_type: str) -> List[AdapterInput]:
        """
        Retrieves a dataset split (train, validation, or test) based on the split type.

        Args:
            split_type (str): The type of dataset split to retrieve. 
                              It can be "train", "val", or "test".

        Returns:
            List[AdapterInput]: A list of AdapterInput objects representing the requested dataset split.
        """
        start, end = self.splits
        list_of_tuples = []

        for target_ts, known_ts, observed_ts in zip(self.target, self.known, self.observed):
            if split_type == "train":
                list_of_tuples.append(
                    AdapterInput(
                        target=target_ts[:start],
                        known=known_ts[:start],
                        observed=observed_ts[:start],
                        static=self.static,
                    )
                )
            elif split_type == "val":
                list_of_tuples.append(
                    AdapterInput(
                        target=target_ts[start:end],
                        known=known_ts[start:end],
                        observed=observed_ts[start:end],
                        static=self.static,
                    )
                )
            elif split_type == "test":
                list_of_tuples.append(
                    AdapterInput(
                        target=target_ts[end:],
                        known=known_ts[end:],
                        observed=observed_ts[end:],
                        static=self.static,
                    )
                )
        return list_of_tuples
    



    def get_train_set(self)-> List[AdapterInput]:
        """
        Retrieves the training dataset.

        Returns:
            List[AdapterInput]: A list of AdapterInput objects representing the training dataset.
        """
        return self._get_split_set("train")

    def get_val_set(self)-> List[AdapterInput]:
        """
        Retrieves the validation dataset.

        Returns:
            List[AdapterInput]: A list of AdapterInput objects representing the validation dataset.
        """
        return self._get_split_set("val")

    def get_test_set(self)-> List[AdapterInput]:
        """
        Retrieves the test dataset.

        Returns:
            List[AdapterInput]: A list of AdapterInput objects representing the test dataset.
        """
        return self._get_split_set("test")
    

