import os
import matplotlib.pyplot as plt
import pandas as pd
from clearml import Task
import logging

from forcateri.reporting.resultreporter import ResultReporter
from ..data.timeseries import TimeSeries
from .metric import Metric
from ..data.adapterinput import AdapterInput
from ..model.modeladapter import ModelAdapter
from typing import List

logger = logging.getLogger(__name__)

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
    
    def _plot_metrics(self, metric_results=None):
        logger.info("Plotting metrics results...")
        clearml_logger = Task.current_task().get_logger()
        if metric_results is None:
            metric_results = self.metric_results

        for model_name, model_metrics in metric_results.items():
            for metric_name, metric_list in model_metrics.items():
                fig, ax = plt.subplots(figsize=(10, 5))
                for i, df in enumerate(metric_list):
                    if isinstance(df, pd.DataFrame):
                        # Skip single-point DataFrames
                        if len(df) <= 1:
                            continue
                        # Dynamically select the second index level for x-axis if possible
                        if (
                            isinstance(df.index, pd.MultiIndex)
                            and len(df.index.names) > 1
                        ):
                            x = df.index.get_level_values(df.index.names[1])
                            xlabel = df.index.names[1]
                        elif isinstance(df.index, pd.MultiIndex):
                            x = df.index.get_level_values(df.index.names[0])
                            xlabel = df.index.names[0]
                        else:
                            x = df.index
                            xlabel = "Index"
                        for col in df.columns:
                            ax.plot(x, df[col], label=f"Test series id: {i} - {col}")
                    else:
                        # Skip plotting for non-DataFrame objects
                        continue

                ax.set_title(f"{metric_name} for {model_name}")
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Metric Value")
                ax.legend()
                plt.tight_layout()
                plt.show()
                # clearml_logger.report_matplotlib_figure(
                # title=f"{metric_name} ({model_name})",
                # series="metrics",
                # figure=fig,
                # iteration=0
                # )
                plt.close()
    
    def _plot_predictions(self):
        super()._plot_predictions()
        #logger.info("Plotting model predictions...")
        # clearml_logger = Task.current_task().get_logger()
        # for model, prediction_ts_list in self.model_predictions.items():
        #     for i, (adapter_input, pred_ts) in enumerate(
        #         zip(self.test_data, prediction_ts_list)
        #     ):
        #         gt_ts = adapter_input.target
        #         offsets = pred_ts.data.index.get_level_values("offset").unique()

        #         logger.debug(
        #             f"Plotting predictions for model {model.__class__.__name__} on test series {i}."
        #         )
        #         if pred_ts.representation == TimeSeries.QUANTILE_REP:
        #             for offset in offsets:
        #                 logger.debug(f"Plotting predictions for offset {offset}.")
        #                 pred_df = pred_ts.by_time(offset).copy()
        #                 gt_df = gt_ts.by_time(horizon=0).loc[pred_df.index]

        #                 # Skip if no data
        #                 if len(pred_df) <= 1:
        #                     continue

        #                 # Flatten MultiIndex columns if needed
        #                 if isinstance(pred_df.columns, pd.MultiIndex):
        #                     pred_df.columns = pred_df.columns.get_level_values(1).astype(
        #                         float
        #                     )

        #                 quantiles = sorted(pred_df.columns.astype(float))
        #                 lower_q = quantiles[0]
        #                 upper_q = quantiles[-1]
        #                 median_q = min(quantiles, key=lambda q: abs(q - 0.5))

        #                 fig, ax = plt.subplots(figsize=(12, 6))

        #                 # --- Plot lower quantile (dashed line)
        #                 ax.plot(
        #                     pred_df.index,
        #                     pred_df[lower_q],
        #                     linestyle="--",
        #                     color="tab:blue",
        #                     alpha=0.7,
        #                     linewidth=1.0,
        #                     label=f"Lower q={lower_q:.2f}",
        #                 )

        #                 # --- Plot upper quantile (dashed line)
        #                 ax.plot(
        #                     pred_df.index,
        #                     pred_df[upper_q],
        #                     linestyle="--",
        #                     color="tab:blue",
        #                     alpha=0.7,
        #                     linewidth=1.0,
        #                     label=f"Upper q={upper_q:.2f}",
        #                 )

        #                 # --- Plot median quantile (solid line)
        #                 if median_q in pred_df.columns:
        #                     ax.plot(
        #                         pred_df.index,
        #                         pred_df[median_q],
        #                         color="tab:blue",
        #                         linewidth=1.5,
        #                         label=f"Median q={median_q:.2f}",
        #                     )

        #                 # --- Plot ground truth (black dashed)
        #                 gt_df.columns = ["Ground Truth"]
        #                 ax.plot(
        #                     gt_df.index,
        #                     gt_df["Ground Truth"],
        #                     color="black",
        #                     linestyle="--",
        #                     linewidth=1.2,
        #                     label="Ground Truth",
        #                 )

        #                 # --- Aesthetics
        #                 ax.set_title(
        #                     f"{model.__class__.__name__} — Test Series {i} — Offset {offset}"
        #                 )
        #                 ax.set_xlabel("Time")
        #                 ax.set_ylabel("Value")
        #                 ax.grid(True, linestyle="--", alpha=0.4)
        #                 ax.legend(loc="upper left", fontsize=9)
        #                 plt.xticks(rotation=30)
        #                 plt.tight_layout()
        #                 plt.show()
        #                 # clearml_logger.report_matplotlib_figure(
        #                 #     title=f"Predictions ({model.__class__.__name__}) - Test Series {i} - Offset {offset}",
        #                 #     series="predictions",
        #                 #     figure=fig,
        #                 #     iteration=0
        #                 # )
        #                 plt.close()