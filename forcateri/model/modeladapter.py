import datetime
from abc import ABC
from pathlib import Path
from typing import Any, List, Optional, Union
import logging
from ..data.adapterinput import AdapterInput
from ..data.timeseries import TimeSeries

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):

    def __init__(self, name: Optional[str] = None):
        # default to the class name
        self._name = name or self.__class__.__name__

    @property
    def name(self):
        return self._name
    
    def fit(
        self,
        train_data: List[AdapterInput],
        val_data: Optional[List[AdapterInput]]=None,
    ):
        raise NotImplementedError(
            "Method not overridden in concrete adapter implementation"
        )

    def predict(
        self,
        data: List[AdapterInput],
        use_rolling_window: bool = True,
    ):
        raise NotImplementedError(
            "Method not overridden in concrete adapter implementation"
        )

    def tune(
        self,
        train_data: List[AdapterInput],
        val_data: Optional[List[AdapterInput]] = None,
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

    def convert_input(self, input: list[AdapterInput]) -> list[Any]:
        target = [self.to_model_format(t.target) for t in input]

        # Known
        known_values = [t.known for t in input]
        if all(v is not None for v in known_values):
            known = [self.to_model_format(v) for v in known_values]
        else:
            known = None
            logger.warning("Some 'known' values are missing; setting known=None")

        # Observed
        observed_values = [t.observed for t in input]
        if all(v is not None for v in observed_values):
            observed = [self.to_model_format(v) for v in observed_values]
        else:
            observed = None
            logger.warning("Some 'observed' values are missing; setting observed=None")

        # Static
        static_values = [t.static for t in input]
        if all(v is not None for v in static_values):
            static = static_values
        else:
            static = None
            logger.warning("Some 'static' values are missing; setting static=None")

        return target, known, observed, static

    def convert_output(self, output: List[Any]) -> List[TimeSeries]:
        """
        Converts the output data into the standardized format.
        """
        logger.debug("Converting model output to standardized TimeSeries format")
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
