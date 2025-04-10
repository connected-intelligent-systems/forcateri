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
        pass
 
    def _fetch_from_cache(self):
        pass

class BaltBestAggregatedAPIData(BaltBestAPIData):
    
    def __init__(self,**kwargs):
        super().__init__(name=kwargs['name'], url=kwargs['url'],local_copy = kwargs['local_copy'])
        self.ts: Optional[List[TimeSeries]] = None

    def get_data(self):
        super().get_data()

    def _fetch_from_cache(self):

        #TODO, discuss the data transformation, interpolations and etc, should it be done here or somewhere else
        df = pd.read_csv(self.local_copy)
        df[self.time_col] = pd.to_datetime(df[self.time_col])
        df = (
            df.set_index(self.time_col)
            .groupby(self.group_col)
            .resample('60min')
            .asfreq()
            .drop(columns=[self.group_col])
            .reset_index())
        self.ts = TimeSeries.from_group_dataframe(df=df,
                                               group_col = self.group_col,
                                               time_col=self.time_col,
                                               value_cols=self.value_cols,
                                               freq = self.freq
                                               )
    def _fetch_data_from_api(self):
        pass






