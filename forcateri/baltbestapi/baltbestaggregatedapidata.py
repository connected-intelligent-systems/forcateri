from typing import List, Optional
from forcateri.baltbestapi.baltbestapidata import BaltBestAPIData
from ..data.timeseries import TimeSeries
import pandas as pd

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
        raise NotImplementedError("Subclasses must implement this method.")

