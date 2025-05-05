from enum import Enum

class SeriesRole(Enum):
    TARGET = "target"
    OBSERVED = "observed"
    KNOWN = "known"
    STATIC = "static"