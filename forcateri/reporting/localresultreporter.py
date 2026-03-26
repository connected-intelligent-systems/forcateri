import os
import pandas as pd
from forcateri.reporting.resultreporter import ResultReporter
import datetime
from .metric import Metric
from ..data.adapterinput import AdapterInput
from ..model.modeladapter import ModelAdapter
from typing import List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LocalResultReporter(ResultReporter):
    METRICS_DIR = "metrics"
    MODELS_DIR = "models"
    PREDICTIONS_DIR = "predictions"
    PLOTS_DIR = "plots"

    def __init__(
        self,
        models: List[ModelAdapter],
        metrics: List[Metric],
        test_data: List[AdapterInput] = None,
        save_path: Path = Path("reports"),
    ):
        super().__init__(models, metrics, test_data=test_data)
        save_path.mkdir(parents=True, exist_ok=True)
        self.save_path = save_path
    def report_all(self):

        super().report_all()
        model_dir = self.save_path / self.MODELS_DIR
        model_dir.mkdir(parents=True, exist_ok=True)
        for model in self.models:
            save_path = model_dir / f"{model.name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            model.save(save_path)
            logger.info(f"Saved model {model.name} to {save_path}")

    def report_metrics(self):
        
        super().report_metrics()

        metrics_dir = self.save_path / self.METRICS_DIR
        metrics_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Reporting metrics to {metrics_dir}...")

        # save each DataFrame separately if there are multiple
        for i, df in enumerate(self.computed_metrics):
            filename = (
                metrics_dir / f"all_metrics_results_{i}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
                if len(self.computed_metrics) > 1
                else metrics_dir / f"all_metrics_results_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
            )
            df.to_csv(filename, index=False)

    def plot_metrics(self):
        plots_dir = self.save_path / self.PLOTS_DIR
        plots_dir.mkdir(parents=True, exist_ok=True)
        super().plot_metrics()
        for fig, model_name, metric_name in self.metric_plots:
            fig.write_html(f"{plots_dir}/{model_name}_{metric_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.html")

    def plot_predictions(self):

        predictions_plot_dir = self.save_path / self.PLOTS_DIR
        predictions_plot_dir.mkdir(parents=True, exist_ok=True)
        super().plot_predictions()
        for fig, model_name, test_idx, offset in self.prediction_plots:
            fig.write_html(
                f"{predictions_plot_dir}/{model_name}_test{test_idx}_offset{offset}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.html"
            )
    
    def report_predictions(self):
        super().report_predictions()
        for model_name, predictions in self.computed_predictions.items():
            pred_dir = self.save_path / self.PREDICTIONS_DIR
            pred_dir.mkdir(parents=True, exist_ok=True)
            filename = pred_dir / f"{model_name}_predictions_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
            with open(filename, "wb") as f:
                pd.to_pickle(predictions, f)
            
            logger.info(f"Saved predictions for model {model_name} to {filename}")
