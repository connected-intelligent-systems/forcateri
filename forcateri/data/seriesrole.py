from enum import Enum


class SeriesRole(Enum):
    TARGET = "TARGET"
    OBSERVED = "OBSERVED"
    KNOWN = "KNOWN"
    STATIC = "STATIC"