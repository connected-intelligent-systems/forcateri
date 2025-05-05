from .timeseries import TimeSeries
from typing import Dict, NamedTuple

class AdapterInput(NamedTuple):
    target: TimeSeries
    observed: TimeSeries
    known: TimeSeries
    static: Dict[str, float]