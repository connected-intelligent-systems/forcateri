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

    def __init__(self, models: List[ModelAdapter], metrics: List[Metric]):
        super().__init__(models, metrics)


    def report_all(self, test_data):

        super().report_all(test_data)
        for model in self.models:
            save_path = Path("models") / f"{model.model_name}.pkl"
            model.save(save_path)
            logger.info(f"Saved model {model.model_name} to {save_path}")

    def report_metrics(self):
        super().report_metrics()
        os.makedirs("reports", exist_ok=True)
        logger.info(f"Metric results: {self.metric_results}")
        for metric_name, model_results in self.metric_results.items():
            all_results = []
            for model_name, result_df_list in model_results.items():
                result = pd.concat(result_df_list, axis=0)
                result["model"] = model_name
                all_results.append(result)

            final_df = pd.concat(all_results, axis=0)
            final_df.reset_index(inplace=True)
            final_df.to_csv(f"reports/{metric_name}_results.csv", index=False)
        with open("reports/local_metric_results.pkl", "wb") as f:
            pickle.dump(self.metric_results, f)



    def _plot_metrics(self, metric_results=None, save_dir="plots"):
        os.makedirs(save_dir,exist_ok=True)
        figures = super()._plot_metrics(metric_results)
        for fig, model_name, metric_name in figures:
            fig.write_html(f"{save_dir}/{model_name}_{metric_name}.html")


    def _plot_predictions(self, save_dir="plots"):
        os.makedirs(save_dir,exist_ok=True)
        figures = super()._plot_predictions()
        for fig, model_name,test_idx,offset in figures:
            fig.write_html(f"{save_dir}/{model_name}_test{test_idx}_offset{offset}.html")
            
