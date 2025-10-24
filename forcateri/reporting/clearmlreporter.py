import os
import matplotlib.pyplot as plt
import pandas as pd
from clearml import Task

from forcateri.reporting.resultreporter import ResultReporter
from .metric import Metric
from ..data.adapterinput import AdapterInput
from ..model.modeladapter import ModelAdapter
from typing import List


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
    
    def _plot_predictions(self):
        """Override to add ClearML-specific plot handling"""
        task = Task.current_task()
        # ClearML-specific implementation
        plot_counter = 0
        
        for model, prediction_ts_list in self.model_predictions.items():
            for i, (adapter_input, pred_ts) in enumerate(
                zip(self.test_data, prediction_ts_list)
            ):
                gt_ts = adapter_input.target
                offsets = pred_ts.data.index.get_level_values("offset").unique()
                
                for offset in offsets:
                    pred_df = pred_ts.by_time(offset).copy()
                    gt_df = gt_ts.by_time(horizon=0).loc[pred_df.index]
                    
                    if isinstance(pred_df.columns, pd.MultiIndex):
                        pred_df.columns = pred_df.columns.get_level_values(1).astype(float)

                    quantiles = sorted(pred_df.columns.astype(float))
                    lower_q = quantiles[0]
                    upper_q = quantiles[-1]
                    median_q = min(quantiles, key=lambda q: abs(q - 0.5))

                    fig, ax = plt.subplots(figsize=(12, 6))

                    # Plot median prediction
                    ax.plot(
                        pred_df.index,
                        pred_df[median_q],
                        label=f"Forecast (q={median_q})",
                        color="blue",
                        linewidth=0.8,
                    )

                    # Plot confidence interval if exists
                    if lower_q != upper_q:
                        ax.fill_between(
                            pred_df.index,
                            pred_df[lower_q],
                            pred_df[upper_q],
                            color="blue",
                            alpha=0.2,
                            label=f"Confidence (q={lower_q}-{upper_q})",
                        )

                    gt_df.columns = ["Ground Truth"]
                    ax.plot(
                        gt_df.index,
                        gt_df["Ground Truth"],
                        label="Ground Truth",
                        color="black",
                        linestyle="--",
                        linewidth=0.8,
                    )

                    title = f"Model {model.__class__.__name__} — Test series id: {i} — Offset: {offset}"
                    ax.set_title(title, fontsize=14)
                    ax.set_xlabel("Time", fontsize=12, weight="bold")
                    ax.set_ylabel("Value", fontsize=12)
                    ax.grid(True, linestyle="--", alpha=0.4)
                    ax.legend(loc="upper left", fontsize=12)
                    plt.xticks(rotation=30)
                    plt.tight_layout()

                    # Report to ClearML
                    task.logger.report_matplotlib_figure(
                        title="Prediction Plots",
                        series=f"{model.__class__.__name__}_Series_{i}_Offset_{offset}",
                        iteration=plot_counter,
                        figure=fig,
                        report_interactive=True  # This makes it interactive!
                    )
                    
                    plot_counter += 1
                    plt.close(fig)  # Close to prevent memory issues