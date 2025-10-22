from abc import abstractmethod
from pathlib import Path
from typing import List

from .datasource import DataSource
from .timeseries import TimeSeries


class CachedAPIData(DataSource):

    @abstractmethod
    def __init__(self):
        super().__init__()
        # local_copy to be dynamically updated after the download.
        self.local_copy: Path = None

    @abstractmethod
    def is_up2date(self):
        # TODO update the logic later
        # self.last_updated = datetime.now()
        # return True
        pass

    @abstractmethod
    def update_local_copy(self):
        # TODO update the logic later
        pass

    @abstractmethod
    def get_data(self) -> List[TimeSeries]:
        pass

    @abstractmethod
    def _fetch_data_from_api(self):

        pass

    @abstractmethod
    def _fetch_from_cache(self):
        pass
