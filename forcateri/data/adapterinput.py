from typing import Dict, NamedTuple, Optional

from .timeseries import TimeSeries


class AdapterInput(NamedTuple):
    target: TimeSeries
    observed: Optional[TimeSeries]
    known: Optional[TimeSeries]
    static: Optional[Dict[str, float]]
