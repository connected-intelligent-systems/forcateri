from dataclasses import dataclass
from typing import Optional, Dict

from .timeseries import TimeSeries


@dataclass
class AdapterInput:
    target: TimeSeries
    observed: Optional[TimeSeries] = None
    known: Optional[TimeSeries] = None
    static: Optional[Dict[str, float]] = None
