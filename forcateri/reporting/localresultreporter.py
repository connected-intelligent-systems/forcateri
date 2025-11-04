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


def save_plots(save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            original_show = plt.show

            def save_show(*s_args, **s_kwargs):
                fig = plt.gcf()
                # get the first axis of the figure
                ax = fig.axes[0] if fig.axes else None
                if ax and ax.get_title():
                    title = ax.get_title()
                else:
                    title = "plot"
                # sanitize filename
                filename = title.replace(" ", "_").replace("/", "_") + ".png"
                filepath = os.path.join(save_dir, filename)
                fig.savefig(filepath)
                plt.close(fig)

            plt.show = save_show
            try:
                return func(*args, **kwargs)
            finally:
                plt.show = original_show

        return wrapper
    return decorator

class LocalResultReporter(ResultReporter):
    def __init__(self, test_data: List[AdapterInput], models: List[ModelAdapter], metrics: List[Metric]):
        super().__init__(test_data, models, metrics)
    
    def report_metrics(self):
        super().report_metrics()
        os.makedirs('reports', exist_ok=True)
        print(self.metric_results)
        for metric_name, model_results in self.metric_results.items():
            all_results = []
            for model_name, result_df_list in model_results.items():
                result = pd.concat(result_df_list, axis=0)
                result['model'] = model_name
                all_results.append(result)

            final_df = pd.concat(all_results, axis=0)
            final_df.reset_index(inplace=True)
            final_df.to_csv(f'reports/{metric_name}_results.csv', index=False)
        with open('reports/local_metric_results.pkl', 'wb') as f:
            pickle.dump(self.metric_results, f)

    @save_plots(save_dir="my_saved_plots")
    def _plot_metrics(self, metric_results=None):
        #TODO Add decorator of saving the plots locally
        return super()._plot_metrics(metric_results)
    
    @save_plots(save_dir="my_saved_plots")
    def _plot_predictions(self):
        return super()._plot_predictions()