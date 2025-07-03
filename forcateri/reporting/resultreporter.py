from typing import List
import matplotlib.pyplot as plt
import seaborn as sns

from .metric import Metric
from ..data.adapterinput import AdapterInput
from ..model.modeladapter import ModelAdapter


class ResultReporter:

    def __init__(self,
        test_data: List[AdapterInput], models: List[ModelAdapter], metrics: List[Metric]
    ):
        self.test_data = test_data
        self.models = models
        self.metrics = metrics

    def report_all(self):
        pass

    def __compute_metrics():
        pass

    def __select_debug_samples():
        pass

    def __report_metrics():
        pass

    def __create_plots(self,df):

        for offset_value, group_df in df.groupby(level='offset'):
            ax = group_df.droplevel('offset').plot(
                marker='o', 
                title=f'Offset: {offset_value}', 
                xlabel='time_stamp', 
                ylabel='Value'
            )
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    def __report_plots():
        pass

    def __report_debug_samples():
        pass

    def __persist_artifacts():
        pass
