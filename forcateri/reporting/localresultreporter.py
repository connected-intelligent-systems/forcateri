import os
import pickle
import pandas as pd
from forcateri.reporting.resultreporter import ResultReporter
from .metric import Metric
from matplotlib import pyplot as plt
from functools import wraps
from ..data.adapterinput import AdapterInput
from ..model.modeladapter import ModelAdapter
from typing import List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LocalResultReporter(ResultReporter):

    def __init__(
        self,
        models: List[ModelAdapter],
        metrics: List[Metric],
        test_data: List[AdapterInput] = None,
    ):
        super().__init__(models, metrics, test_data=test_data)

    def report_all(self):

        super().report_all()
        for model in self.models:
            save_path = Path("models") / f"{model.name}"
            model.save(save_path)
            logger.info(f"Saved model {model.name} to {save_path}")

    def report_metrics(self, save_dir="reports"):
        # call parent, which now returns a list of DataFrames
        super().report_metrics()

        os.makedirs(save_dir, exist_ok=True)
        logger.info(f"Reporting metrics to {save_dir}...")

        # save each DataFrame separately if there are multiple
        for i, df in enumerate(self.computed_metrics):
            filename = (
                f"{save_dir}/all_metrics_results_{i}.csv"
                if len(self.computed_metrics) > 1
                else f"{save_dir}/all_metrics_results.csv"
            )
            df.to_csv(filename, index=False)

    def plot_metrics(self, save_dir="plots"):
        os.makedirs(save_dir, exist_ok=True)
        super().plot_metrics()
        for fig, model_name, metric_name in self.metric_plots:
            fig.write_html(f"{save_dir}/{model_name}_{metric_name}.html")

    def plot_predictions(self, save_dir="plots"):

        os.makedirs(save_dir, exist_ok=True)
        super().plot_predictions()
        for fig, model_name, test_idx, offset in self.prediction_plots:
            fig.write_html(
                f"{save_dir}/{model_name}_test{test_idx}_offset{offset}.html"
            )
