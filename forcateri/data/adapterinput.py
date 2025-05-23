from typing import Dict, NamedTuple

from .timeseries import TimeSeries


class AdapterInput(NamedTuple):
    target: TimeSeries
    observed: TimeSeries
    known: TimeSeries
    static: Dict[str, float]
