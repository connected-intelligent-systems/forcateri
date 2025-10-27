import os
import pickle
from forcateri.reporting.resultreporter import ResultReporter
from .metric import Metric
from ..data.adapterinput import AdapterInput
from ..model.modeladapter import ModelAdapter
from typing import List

class LocalResultReporter(ResultReporter):
    def __init__(self, test_data: List[AdapterInput], models: List[ModelAdapter], metrics: List[Metric]):
        super().__init__(test_data, models, metrics)
    
    def report_metrics(self):
        super().report_metrics()
        os.makedirs('reports', exist_ok=True)
        with open('reports/local_metric_results.pkl', 'wb') as f:
            pickle.dump(self.metric_results, f)

    def _plot_metrics(self, metric_results=None):
        #TODO Add decorator of saving the plots locally
        return super()._plot_metrics(metric_results)