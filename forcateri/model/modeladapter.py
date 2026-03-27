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
        """
        Converts the input data into the model-specific format.
        """
        logger.debug("Converting input data to model-specific format")
        def process_input(covariate:List[TimeSeries])->list[Any]:
            if all(v is not None for v in covariate):
                return [self.to_model_format(v) for v in covariate]
            elif any(v is not None for v in covariate):
                logger.warning("Some covariate values are missing; Please make sure that covariates are present along with target for all input samples. Setting this covariate to None for this batch.")
                return None
            else:
                logger.info("No covariate values provided; setting this covariate to None")
                return None
        target = [self.to_model_format(t.target) for t in input]
        known = process_input([t.known for t in input])
        observed = process_input([t.observed for t in input])
        static = process_input([t.static for t in input])
        return target, known, observed, static
            
    # def convert_input(self, input: list[AdapterInput]) -> list[Any]:
    #     target = [self.to_model_format(t.target) for t in input]

    #     # Known
    #     known_values = [t.known for t in input]
    #     if all(v is not None for v in known_values):
    #         known = [self.to_model_format(v) for v in known_values]
    #     elif any(v is not None for v in known_values):
    #         known = None
    #         logger.warning("Some 'known' covariate values are missing; Please make sure that covariates are present along with target for all input samples. Setting known=None for this batch.")
    #     else:
    #         known = None
    #         logger.info("No 'known' covariate values provided; setting known=None")

    #     # Observed
    #     observed_values = [t.observed for t in input]
    #     if all(v is not None for v in observed_values):
    #         observed = [self.to_model_format(v) for v in observed_values]
    #     elif any(v is not None for v in observed_values):
    #         observed = None
    #         logger.warning("Some 'observed' covariate values are missing; Please make sure that covariates are present along with target for all input samples. Setting observed=None for this batch.")
    #     else:
    #         observed = None
    #         logger.info("No 'observed' covariate values provided; setting observed=None")

    #     # Static
    #     static_values = [t.static for t in input]
    #     if all(v is not None for v in static_values):
    #         static = static_values
    #     elif any(v is not None for v in static_values):
    #         static = None
    #         logger.warning("Some 'static' covariate values are missing; Please make sure that covariates are present along with target for all input samples. Setting static=None for this batch.")
    #     else:
    #         static = None
    #         logger.info("No 'static' covariate values provided; setting static=None")

    #     return target, known, observed, static

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
