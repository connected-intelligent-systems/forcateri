import os
import matplotlib.pyplot as plt
import pandas as pd
from clearml import Task
from functools import wraps

from forcateri.reporting.resultreporter import ResultReporter
from .metric import Metric
from ..data.adapterinput import AdapterInput
from ..model.modeladapter import ModelAdapter
from typing import List

def clearml_log_plot(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Run the original plotting function
        result = func(self, *args, **kwargs)

        # Get ClearML logger
        task = Task.current_task()
        if task:
            logger = task.get_logger()

            # Collect all open figures
            figs = [plt.figure(n) for n in plt.get_fignums()]

            for i, fig in enumerate(figs):
                logger.report_matplotlib(
                    title=f"{func.__name__}_{i}",
                    series=func.__name__,
                    figure=fig,
                    iteration=0,
                )

            # Close figures after logging
            for fig in figs:
                plt.close(fig)

        return result
    return wrapper

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
        for metric_name, model_results in self.metric_results.items():
            all_results = []
            for model_name, result_df_list in model_results.items():
                result = pd.concat(result_df_list, axis=0)
                result['model'] = model_name
                all_results.append(result)

            final_df = pd.concat(all_results, axis=0)
            final_df.reset_index(inplace=True)

            final_df.to_csv(f'{metric_name}_results.csv', index=False)
            Task.current_task().upload_artifact(name=f'{metric_name}_results.csv', artifact_object=final_df)
    
    @clearml_log_plot
    def _plot_metrics(self, metric_results=None):
        super()._plot_metrics(metric_results)
    
    @clearml_log_plot
    def _plot_predictions(self):
        super()._plot_predictions()