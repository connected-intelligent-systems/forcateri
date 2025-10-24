import os
import matplotlib.pyplot as plt
import pandas as pd
from clearml import Task

from forcateri.reporting.resultreporter import ResultReporter
from .metric import Metric
from ..data.adapterinput import AdapterInput
from ..model.modeladapter import ModelAdapter
from typing import List


class ClearMLReporter(ResultReporter):

    def __init__(
        self,
        test_data: List[AdapterInput],
        models: List[ModelAdapter],
        metrics: List[Metric],
    ):
        super().__init__(test_data, models, metrics)

    def report_all(self):
        super().report_all()
        print(f"Test of the metric results {self.metric_results}")
        Task.current_task().upload_artifact(name='Report', artifact_object=self.metric_results)
    
