import datetime
from abc import ABC
from pathlib import Path
from typing import Any, List, Optional, Union

from ..data.adapterinput import AdapterInput
from ..data.timeseries import TimeSeries


class ModelAdapter(ABC):
    
    def __init__(self, model_name: Optional[str] = None):
        # default to the class name
        self.model_name = model_name or self.__class__.__name__
    def fit(
        self,
        train_data: List[AdapterInput],
        val_data: Optional[List[AdapterInput]],
        **kwargs,
    ):
        raise NotImplementedError(
            "Method not overridden in concrete adapter implementation"
        )

    def predict(self, data: List[AdapterInput]):
        raise NotImplementedError(
            "Method not overridden in concrete adapter implementation"
        )

    def tune(
        self,
        train_data: List[AdapterInput],
        val_data: Optional[List[AdapterInput]],
        **kwargs,
    ):
        raise NotImplementedError(
            "Method not overridden in concrete adapter implementation"
        )

    def load(self, path: Union[Path, str]) -> None:
        raise NotImplementedError(
            "Method not overridden in concrete adapter implementation"
        )

    def save(self, path: Union[Path, str]) -> None:
        raise NotImplementedError(
            "Method not overridden in concrete adapter implementation"
        )

    def to_model_format(self, ts: TimeSeries) -> Any:
        """
        Applies model-specific transformations to the time series data.
        """
        raise NotImplementedError(
            "Method not overridden in concrete adapter implementation"
        )

    def convert_input(self, input: List[AdapterInput]) -> List[Any]:
        """
        Converts the input data into the format required by the model.
        """
        target = [self.to_model_format(t.target) for t in input]
        known = [self.to_model_format(t.known) for t in input]
        observed = [self.to_model_format(t.observed) for t in input]
        static = [t.static for t in input]

        return target, known, observed, static

    def convert_output(self, output: List[Any]) -> List[TimeSeries]:
        """
        Converts the output data into the standardized format.
        """
        return [self.to_time_series(o) for o in output]

    def to_time_series(self, ts: Any) -> TimeSeries:
        """
        Converts the model-specific data into the standardized TimeSeries format e.g., inverse scaling.
        """
        raise NotImplementedError(
            "Method not overridden in concrete adapter implementation"
        )

    @classmethod
    def _default_save_path(cls) -> str:
        return f"{cls.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"