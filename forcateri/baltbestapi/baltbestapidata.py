from ..data.cachedapidata import CachedAPIData
from ..data.timeseries import TimeSeries
import pandas as pd
from typing import List, Optional

class BaltBestAPIData(CachedAPIData):
    
    def __init__(self,
                 group_col:str='room_id',
                 time_col:str='datetime',
                 freq:str='h',
                 value_cols:List[str]=['q_hca','temperature_1_max','temperature_2_max','temperature_outdoor_avg', 'temperature_room_avg'],
                 **kwargs):
        super().__init__(name=kwargs['name'])
        self.url = kwargs['url']
        self.local_copy = kwargs['local_copy']
        self.group_col = group_col
        self.time_col = time_col
        self.value_cols = value_cols
        self.freq = freq
    def get_data(self):

        super().get_data()


    def _fetch_data_from_api(self):
        #TODO For now, the logic does not get the data from API
        raise NotImplementedError("Subclasses must implement this method.")
 
    def _fetch_from_cache(self):
        raise NotImplementedError("Subclasses must implement this method.")








