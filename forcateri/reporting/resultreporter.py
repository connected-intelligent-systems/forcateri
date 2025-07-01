from typing import List
import matplotlib.pyplot as plt

from .metric import Metric
from ..data.adapterinput import AdapterInput
from ..model.modeladapter import ModelAdapter


class ResultReporter:

    def __init__(
        test_data: List[AdapterInput], models: List[ModelAdapter], metrics: List[Metric]
    ):
        pass

    def report_all(self):
        pass

    def __compute_metrics():
        pass

    def __select_debug_samples():
        pass

    def __report_metrics():
        pass

    def __create_plots():
        pass

    def __report_plots():
        pass

    def __report_debug_samples():
        pass

    def __persist_artifacts():
        pass
