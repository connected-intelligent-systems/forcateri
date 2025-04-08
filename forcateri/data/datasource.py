from abc import ABC,abstractmethod
from typing import List
from .timeseries import TimeSeries
from datetime import datetime
from typing import Optional

class DataSource(ABC):
    """
    The base class for data sources
    """
    def __init__(self,**kwargs):
        self.name = kwargs['name']
        self.source_type = kwargs['source_type']
        self.last_updated:Optional[datetime] = None
        
    @abstractmethod
    def get_data(self) -> List[TimeSeries]:
        """Retrieve the data as a list of TimeSeries objects."""
        pass 