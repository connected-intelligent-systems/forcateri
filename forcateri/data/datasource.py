from abc import ABC,abstractmethod
from typing import List,Optional
from .timeseries import TimeSeries
from datetime import datetime


class DataSource(ABC):
    """
    The base class for data sources
    """
    @abstractmethod
    def __init__(self,**kwargs):
        self.last_updated:Optional[datetime] = None
        
    @abstractmethod
    def get_data(self) -> List[TimeSeries]:
        """Retrieve the data as a list of TimeSeries objects."""
        pass 