from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from .timeseries import TimeSeries


class DataSource(ABC):
    """
    The base class for data sources
    """

    @abstractmethod
    def get_data(self) -> List[TimeSeries]:
        """Retrieve the data as a list of TimeSeries objects."""
        pass
