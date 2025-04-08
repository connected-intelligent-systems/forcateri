from ..data.cachedapidata import CachedAPIData

class BaltBestAPIData(CachedAPIData):
    
    def __init__(self,**kwargs):
        super().__init__(name=kwargs['name'])
        self.url = kwargs['url']
    def _fetch_data_from_api(self):
        pass 
    def _fetch_from_cache(self):
        pass

class BaltBestAggregatedAPIData(BaltBestAPIData):
    pass



