import datetime
from abc import ABC, ABCMeta, abstractmethod
from pathlib import Path
from ..data.timeseries import TimeSeries
from typing import List,Optional, Any, Union
import logging
import pickle
from ..model.modelexceptions import ModelNotFittedError
from ..data.adapterinput import AdapterInput


class ModelAdapter(ABC):
    @abstractmethod
    def __init__(self, *args,**kwargs):
        pass
    
    @abstractmethod
    def fit(self, train_data:List[AdapterInput],val_data:Optional[List[AdapterInput]], **kwargs):
        pass
        

    @abstractmethod
    def predict(self, data:List[AdapterInput]):
        if self.target is None:
            logging.error("Predict called befor the model was fitted")
            raise ModelNotFittedError("The model must be fitted before predicting.")

    @abstractmethod
    def tune(self,train_data:List[AdapterInput], val_data:Optional[List[AdapterInput]], **kwargs):
        pass 
    
    @abstractmethod
    def load(self, path: Union[Path, str]) -> None:
        pass

    @abstractmethod
    def save(self,path: Union[Path, str]) -> None:
        pass

        


    @abstractmethod
    def to_model_format(ts:TimeSeries) -> Any:
        """
        Applies model-specific transformations to the time series data.
        """
        pass 

    @abstractmethod
    def to_time_series(ts:Any) -> TimeSeries:
        """
        Converts the model-specific data into the standardized TimeSeries format e.g., inverse scaling.
        """
        pass

    @classmethod
    def _default_save_path(cls) -> str:
        return f"{cls.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
