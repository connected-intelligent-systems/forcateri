from .datasource import DataSource
from typing import List
from pathlib import Path
from abc import abstractmethod
from datetime import datetime
from .timeseries import TimeSeries

class CachedAPIData(DataSource):
    
    def __init__(self,**kwargs):
        super().__init__(name=kwargs['name'],source_type="cached_api")
        #local_copy to be dynamically updated after the download.
        self.local_copy: Path = None
        
    @abstractmethod
    def is_up2date(self):
        #TODO update the logic later
        self.last_updated = datetime.now()
        return True
    
    @abstractmethod
    def update_local_copy(self):
        #TODO update the logic later
        pass

    @abstractmethod
    def get_data(self) -> List[TimeSeries]:

        if self.local_copy:
            if self.is_up2date():
                self._fetch_from_cache()
            else:
                self.update_local_copy()
        else:
            self._fetch_data_from_api()

    @abstractmethod
    def _fetch_data_from_api(self):
        #Here the local_copy variable to be initialized
        pass
    @abstractmethod
    def _fetch_from_cache(self):
        pass