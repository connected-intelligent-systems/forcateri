import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Union

from forcateri.data import adapterinput

from ..data.adapterinput import AdapterInput
from ..data.timeseries import TimeSeries


class ModelAdapter(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(
        self,
        train_data: List[AdapterInput],
        val_data: Optional[List[AdapterInput]],
        **kwargs,
    ):
        raise NotImplementedError(
            "Method not overridden in concrete adapter implementation"
        )

    @abstractmethod
    def predict(self, data: List[AdapterInput]):
        raise NotImplementedError(
            "Method not overridden in concrete adapter implementation"
        )

    @abstractmethod
    def tune(
        self,
        train_data: List[AdapterInput],
        val_data: Optional[List[AdapterInput]],
        **kwargs,
    ):
        raise NotImplementedError(
            "Method not overridden in concrete adapter implementation"
        )

    @abstractmethod
    def load(self, path: Union[Path, str]) -> None:
        raise NotImplementedError(
            "Method not overridden in concrete adapter implementation"
        )

    @abstractmethod
    def save(self, path: Union[Path, str]) -> None:
        raise NotImplementedError(
            "Method not overridden in concrete adapter implementation"
        )

    @abstractmethod
    def to_model_format(self, ts: TimeSeries) -> Any:
        """
        Applies model-specific transformations to the time series data.
        """
        raise NotImplementedError(
            "Method not overridden in concrete adapter implementation"
        )

    def convert_input(self, input: List[AdapterInput]) -> Any:
        """
        Converts the input data into the standardized format.
        """
        return [
            AdapterInput(
                target=self.to_model_format(i.target),
                known=self.to_model_format(i.known),
                observed=self.to_model_format(i.observed),
                static=i.static,
            )
            for i in input
        ]

    @abstractmethod
    def to_time_series(ts: Any) -> TimeSeries:
        """
        Converts the model-specific data into the standardized TimeSeries format e.g., inverse scaling.
        """
        raise NotImplementedError(
            "Method not overridden in concrete adapter implementation"
        )

    @classmethod
    def _default_save_path(cls) -> str:
        return f"{cls.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
