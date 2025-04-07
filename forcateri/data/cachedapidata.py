from datasource import DataSource
from typing import List
from pathlib import Path
from abc import abstractmethod
from timeseries import TimeSeries

class CachedAPIData(DataSource):
    
    def __init__(self,name:str):
        super.__init__(name=name,source_type="cached_api")
        #local_copy to be dynamically updated after the download.
        self.local_copy: Path = None

    def is_up2date():
        #TODO update the logic later
        return True
    def update_local_copy():
        #TODO update the logic later
        pass
    def get_data(self) -> List[TimeSeries]:
        if self.local_copy and self.is_up2date():
            self._fetch_from_cache()
        else:
            self._fetch_data_from_api()

    @abstractmethod
    def _fetch_data_from_api(self):
        #Here the local_copy variable to be initialized
        pass
    @abstractmethod
    def _fetch_from_cache(self):
        pass