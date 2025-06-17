from clearml import Task 
from .baltbestaggregatedapidata import BaltBestAggregatedAPIData 

project_name = "ForeSightNEXT/BaltBest"
baltbest = BaltBestAggregatedAPIData(
    name='test', 
    url="baltbest_url", 
    #local_copy="/home/dior00002/dfki/forcateri/_data/hourly_data.csv",
    local_copy = None,
    target = 'delta',
    group_col = 'room_id',
    time_col = 'rounded_ts',
    known = 'outside_temp',
    observed = ['max_temperature_1', 'max_temperature_2','room_temperature'],
    dataset_project='ForeSightNEXT/BaltBest',       # <-- Required for ClearmlDataMixin
    dataset_name='BaltBest_excerpt_12_11_24',          # <-- Required for ClearmlDataMixin
    dataset_version='1.0.0'               # <-- Required for ClearmlDataMixin
    )
data_root = baltbest.get_data()
print(data_root)