from abc import ABC,abstractmethod
from typing import List
from timeseries import TimeSeries

class DataSource(ABC):
    """
    The base class for data sources
    """
    @abstractmethod
    def __init__(self, name:str, source_type:str):
        self.name = name 
        self.source_type = source_type
        self.last_updated = None
        
    @abstractmethod
    def get_data(self) -> List[TimeSeries]:
        """Retrieve the data as a list of TimeSeries objects."""
        pass 