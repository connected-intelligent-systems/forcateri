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
        test_data: List[AdapterInput] = None,
    ):
        super().__init__(models, metrics, test_data=test_data)

    def report_all(self):
        super().report_all()
        print(f"Test of the metric results {self.metric_results}")
        Task.current_task().upload_artifact(
            name="Report", artifact_object=self.metric_results
        )
        Task.current_task().upload_artifact(
            name="Model Predictions", artifact_object=self.model_predictions
        )
        for model in self.models:
            Task.current_task().upload_artifact(
                name=model.model_name, artifact_object=model
            )

    def report_metrics(self):
        super().report_metrics()
        for i, df in enumerate(self.reported_metrics):
            filename = f"metric_results_{i}.csv"
            df.to_csv(filename)
            Task.current_task().upload_artifact(name=filename, artifact_object=filename)

    def plot_metrics(self):

        super().plot_metrics()

        for fig, model_name, metric_name in self.metric_figures:
            filename = f"{model_name}_{metric_name}.html"
            fig.write_html(filename)
            Task.current_task().upload_artifact(name=filename, artifact_object=filename)

    def plot_predictions(self):
        
        super().plot_predictions()
        for fig, model_name, test_idx, offset in self.prediction_figures:
            filename = f"{model_name}_test{test_idx}_offset{offset}.html"
            fig.write_html(filename)
            Task.current_task().upload_artifact(name=filename, artifact_object=filename)
