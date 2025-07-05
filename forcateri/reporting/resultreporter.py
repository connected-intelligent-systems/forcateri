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

    def report_all(self,df):
        self.__create_plots(df)

    def __compute_metrics():
        pass

    def __select_debug_samples():
        pass

    def __report_metrics():
        pass

    def __create_plots(self):

        for model in self.models:
            predictions = model.predict(self.test_data)
            predictions_ts_list = model.to_time_series(predictions)  # List of TimeSeries objects

            for i, (adapter_input, pred_ts) in enumerate(zip(self.test_data, predictions_ts_list)):
                gt_ts = adapter_input.target  # Also a TimeSeries object

                
                offsets = pred_ts.data.index.get_level_values("offset").unique()

                for offset in offsets:
                    pred_df = pred_ts.by_time(offset)
                    gt_df = gt_ts.by_time(offset)

                    # Align on timestamps
                    pred_df_aligned, gt_df_aligned = pred_df.align(gt_df, join='inner')

                    if pred_df_aligned.empty:
                        continue  # No overlapping timestamps

                    ax = pred_df_aligned.plot(label='Prediction', title=f"Sample {i}, Model: {model.name}, Offset: {offset}")
                    gt_df_aligned.plot(ax=ax, label='Ground Truth', linestyle='--', alpha=0.7)
                    ax.legend()
        
    def __report_plots():
        pass

    def __report_debug_samples():
        pass

    def __persist_artifacts():
        pass
