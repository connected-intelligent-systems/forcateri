from datetime import datetime
from typing import Dict, List, NamedTuple, Tuple, Union, Callable

import pandas as pd
import logging
from .adapterinput import AdapterInput
from .datasource import DataSource
from .seriesrole import SeriesRole
from .timeseries import TimeSeries

logger = logging.getLogger(__name__)

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
        roles: List[Dict[SeriesRole, List[str]]],
        splits: Union[Cutoff, List[Cutoff]] = (1.0 / 3.0, 2.0 / 3.0),
    ):
        """
        Initializes the DataProvider.

        Args:
            data_sources (List[DataSource]):
                Data sources used to load time series data.
            roles (Dict[SeriesRole, List[str]]):
                Mapping from series roles to series names.
                Example:
                    {
                        SeriesRole.TARGET: ["target_series"],
                        SeriesRole.KNOWN: ["known_1", "known_2"],
                        SeriesRole.OBSERVED: ["observed_1"]
                    }
            splits (Union[Cutoff, List[Cutoff]], optional):
                Cutoff(s) used to split the data (e.g., train/val/test).
                Defaults to (1/3, 2/3).

        Attributes:
            data_sources (List[DataSource]):
                Provided data sources.
            roles (Dict[SeriesRole, List[str]]):
                Series role definitions.
            splits (Union[Cutoff, List[Cutoff]]):
                Split points for data division.
            target (List):
                Loaded target time series.
            known (List):
                Loaded known covariate series.
            observed (List):
                Loaded observed covariate series.
            static (Dict[str, float]):
                Static features shared across series.
        """

        self.roles = roles
        self.splits = splits
        self.data_sources = data_sources
        self.target = []
        self.known = []
        self.observed = []
        self.static: Dict[str, float] = None
        self.is_separated = False
        # self._separate_ts()

    def _separate_ts(self):
        """ ""
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
        logger.debug(
            "Separating time series data into target, known, and observed categories."
        )

        for data_source, role in zip(self.data_sources, self.roles):
            logger.debug(f"Processing data source: {data_source} with roles: {role}")
            columns_observed = role.get(SeriesRole.OBSERVED) or []
            columns_observed = (
                columns_observed
                if isinstance(columns_observed, list)
                else [columns_observed]
            )

            columns_target = role.get(SeriesRole.TARGET) or []
            columns_target = (
                columns_target if isinstance(columns_target, list) else [columns_target]
            )

            columns_known = role.get(SeriesRole.KNOWN) or []
            columns_known = (
                columns_known if isinstance(columns_known, list) else [columns_known]
            )

            logger.debug(
                f"Identified columns - Target: {columns_target}, Known: {columns_known}, Observed: {columns_observed}"
            )
            data_list = data_source.get_data()
            for ts_obj in data_list:
                self.target.append(ts_obj.get_feature_slice(index=columns_target))
                #self.target.append(ts_obj.get_feature_slice(index=["target"]))
                self.known.append(
                    ts_obj.get_feature_slice(index=columns_known)
                    if len(columns_known) > 0
                    else None
                )
                self.observed.append(
                    ts_obj.get_feature_slice(index=columns_observed)
                    if len(columns_observed) > 0
                    else None
                )

    def _get_split_set(self, split_type: str) -> List[AdapterInput]:
        """
        Retrieves a dataset split (train, validation, or test) based on the split type.

        Args:
            split_type (str): The type of dataset split to retrieve.
                              It can be "train", "val", or "test".

        Returns:
            List[AdapterInput]: A list of AdapterInput objects representing the requested dataset split.
        """
        logger.debug(f"Retrieving {split_type} dataset split.")
        if isinstance(self.splits, tuple):
            start, end = self.splits
            list_of_tuples = []

            for target_ts, known_ts, observed_ts in zip(
                self.target, self.known, self.observed
            ):
                if split_type == "train":
                    logger.debug(
                        "Processing training split. List[AdapterInput] length: %d",
                        len(list_of_tuples),
                    )
                    list_of_tuples.append(
                        AdapterInput(
                            target=target_ts[:start] if target_ts is not None else None,
                            known=known_ts[:start] if known_ts is not None else None,
                            observed=(
                                observed_ts[:start] if observed_ts is not None else None
                            ),
                            static=self.static,
                        )
                    )
                elif split_type == "val":
                    logger.debug(
                        "Processing validation split. List[AdapterInput] length: %d",
                        len(list_of_tuples),
                    )
                    list_of_tuples.append(
                        AdapterInput(
                            target=(
                                target_ts[start:end] if target_ts is not None else None
                            ),
                            known=known_ts[start:end] if known_ts is not None else None,
                            observed=(
                                observed_ts[start:end]
                                if observed_ts is not None
                                else None
                            ),
                            static=self.static,
                        )
                    )
                elif split_type == "test":
                    logger.debug(
                        "Processing test split. List[AdapterInput] length: %d",
                        len(list_of_tuples),
                    )
                    list_of_tuples.append(
                        AdapterInput(
                            target=target_ts[end:] if target_ts is not None else None,
                            known=known_ts[end:] if known_ts is not None else None,
                            observed=(
                                observed_ts[end:] if observed_ts is not None else None
                            ),
                            static=self.static,
                        )
                    )

        return list_of_tuples

    def get_train_set(self) -> List[AdapterInput]:
        """
        Retrieves the training dataset.

        Returns:
            List[AdapterInput]: A list of AdapterInput objects representing the training dataset.
        """
        if not self.is_separated:
            self._separate_ts()
            self.is_separated = True
        return self._get_split_set("train")

    def get_val_set(self) -> List[AdapterInput]:
        """
        Retrieves the validation dataset.

        Returns:
            List[AdapterInput]: A list of AdapterInput objects representing the validation dataset.
        """
        if not self.is_separated:
            self._separate_ts()
            self.is_separated = True
        return self._get_split_set("val")

    def get_test_set(self) -> List[AdapterInput]:
        """
        Retrieves the test dataset.

        Returns:
            List[AdapterInput]: A list of AdapterInput objects representing the test dataset.
        """
        if not self.is_separated:
            self._separate_ts()
            self.is_separated = True
        return self._get_split_set("test")
