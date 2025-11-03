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
    
    def report_metrics(self):
        super().report_metrics()
        for model_name, model_results in self.metric_results.items():
            for metric_name, result_df_list in model_results.items():
                result = pd.concat(result_df_list, axis=0)
                result['model'] = model_name
                result.reset_index(inplace=True)
                csv_name = f'{model_name}_{metric_name}_results.csv'
                result.to_csv(csv_name, index=False)
                Task.current_task().upload_artifact(name=csv_name, artifact_object=csv_name)