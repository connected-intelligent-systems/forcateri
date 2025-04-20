import datetime
from abc import ABC, ABCMeta, abstractmethod
from pathlib import Path
from ..data.timeseries import TimeSeries
from typing import List,Optional, Any, Union
import logging
import pickle

class ModelAdapterError(Exception):
    """Base exception for errors in the ModelAdapter."""
    pass

class ModelNotFittedError(ModelAdapterError):
    """Raised when an operation is attempted that requires a fitted model."""
    pass

class InvalidTimeSeriesError(ModelAdapterError):
    """Raised when an invalid TimeSeries object is provided."""
    pass
class InvalidModelTypeError(ModelAdapterError):
    '''Raised when invalid model is instanciated'''
    pass

class ModelAdapter(ABC):
    @abstractmethod
    def __init__(self, *args,**kwargs):
        self.target:Optional[TimeSeries] = None 
        self.known:Optional[TimeSeries] = None 
        self.observed:Optional[TimeSeries] = None 
        self.static:Optional[TimeSeries] = None
    
    @abstractmethod
    def fit(self, **kwargs):
        # try:
        #     transformed_ts = self.to_model_format(kwargs['target'])
        #     self.target = transformed_ts 
        # except:
        #     raise InvalidTimeSeriesError("The time series cannot be transformed to model's timeseries format")
        pass
        

    @abstractmethod
    def predict(self):
        if self.target is None:
            logging.error("Predict called befor the model was fitted")
            raise ModelNotFittedError("The model must be fitted before predicting.")

    @abstractmethod
    def tune(self):
        pass 
    
    @abstractmethod
    def load(self, path: Union[Path, str]) -> None:
        pass

    @abstractmethod
    def save(self,path: Union[Path, str]) -> None:
        
        pass
        '''
        if path is None:
            path = self._default_save_path + ".pkl"
        path = Path(path)
        try:
            path.parent.mkdir(parents=True,exist_ok=True)
            with open(path,'wb') as f:
                pickle.dump(self,f)
            logging.info(f"model saved to {path}")
        except Exception as e:
            logging.error(f"Failed to save model to {path}: {e}")
            raise IOError(f"Could not save model: {e}")
        '''


    @abstractmethod
    def to_model_format(ts:TimeSeries) -> Any:
        pass 

    @abstractmethod
    def to_time_series(ts:Any) -> TimeSeries:
        pass

    @classmethod
    def _default_save_path(cls) -> str:
        return f"{cls.__name__}_{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}"
