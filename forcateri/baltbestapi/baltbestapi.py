from ..data.cachedapidata import CachedAPIData
from ..data.timeseries import TimeSeries
import pandas as pd

class BaltBestAPIData(CachedAPIData):
    
    def __init__(self,**kwargs):
        super().__init__(name=kwargs['name'])
        self.url = kwargs['url']
        self.local_copy = kwargs['local_copy']

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
        

    def get_data(self):
        super().get_data()

    def _fetch_from_cache(self):

        #TODO, discuss the data transformation, interpolations and etc, should it be done here or somewhere else
        return TimeSeries.from_group_dataframe(df=pd.read_csv(self.local_copy),
                                               group_col = 'room_id',
                                               time_col='datetime',
                                               value_cols=['q_hca','temperature_1_max','temperature_2_max','temperature_outdoor_avg', 'temperature_room_avg']
                                               )
    def _fetch_data_from_api(self):
        pass



