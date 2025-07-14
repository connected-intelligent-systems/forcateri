from typing import Callable, List, OrderedDict
import numpy as np

from .metric import Metric


class DimwiseAggregatedMetric(Metric):

    def __init__(reductions: OrderedDict[List[str], Callable]):
        pass


