import os
import matplotlib.pyplot as plt
import pandas as pd
from clearml import Task
import logging
from functools import wraps
import plotly.io as pio
from plotly.tools import mpl_to_plotly

from forcateri.reporting.resultreporter import ResultReporter
from ..data.timeseries import TimeSeries
from .metric import Metric
from ..data.adapterinput import AdapterInput
from ..model.modeladapter import ModelAdapter
from typing import List
import os
import matplotlib.pyplot as plt
from clearml import Task

logger = logging.getLogger(__name__)

class ClearMLReporter(ResultReporter):

    def __init__(
        self,
        models: List[ModelAdapter],
        metrics: List[Metric],
    ):
        super().__init__(models, metrics)

    def report_all(self, test_data: List[AdapterInput]):
        super().report_all(test_data)
        print(f"Test of the metric results {self.metric_results}")
        Task.current_task().upload_artifact(
            name="Report", artifact_object=self.metric_results
        )
        Task.current_task().upload_artifact(
            name="Model Predictions", artifact_object=self.model_predictions
        )
        for model in self.models:
            Task.current_task().upload_artifact(
                name=model.model_name,
                artifact_object=model
        )

    def report_metrics(self):
        pivot_df = super().report_metrics()
        Task.current_task().upload_artifact(
            name="Metric results all", artifact_object=pivot_df
        )


    
    
    def _plot_metrics(self, metric_results=None):

        figures = super()._plot_metrics(metric_results)

        for fig, model_name, metric_name in figures:
            filename = f"{model_name}_{metric_name}.html"
            fig.write_html(filename)
            Task.current_task().upload_artifact(
                name=filename,
                artifact_object=filename
            )
        
    def _plot_predictions(self):
        figures = super()._plot_predictions()
        for fig, model_name,test_idx,offset in figures:
            filename = f"{model_name}_test{test_idx}_offset{offset}.html"
            fig.write_html(filename)
            Task.current_task().upload_artifact(
                name = filename,
                artifact_object = filename
            )