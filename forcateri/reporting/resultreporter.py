from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .metric import Metric
from ..data.adapterinput import AdapterInput
from ..model.modeladapter import ModelAdapter


class ResultReporter:

    def __init__(
        self,
        test_data: List[AdapterInput],
        models: List[ModelAdapter],
        metrics: List[Metric],
    ):
        self.test_data = test_data
        self.models = models
        self.metrics = metrics

    def report_all(
        self,
    ):  # dont forget to remove predictions after testing
        self._make_predictions()
        self._create_plots()

    def _compute_metrics():
        pass

    def _select_debug_samples():
        pass

    def _report_metrics():
        pass
    
    def _make_predictions(self):
        for model in self.models:
            self.predictions_ts_list = model.predict(self.test_data) 

    def _create_plots(self):
        for i, (adapter_input, pred_ts) in enumerate(zip(self.test_data, self.predictions_ts_list)):
            gt_ts = adapter_input.target  # TimeSeries object
            offsets = pred_ts.data.index.get_level_values("offset").unique()
            for offset in offsets:
                pred_df = pred_ts.by_time(offset).copy()
                
                gt_df = gt_ts.by_time(horizon=0).loc[pred_df.index]  # Align indices
                # Flatten MultiIndex columns if needed
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
                        label=f"Confidence (q={lower_q}-{upper_q})"
                    )

                gt_df.columns = ['Ground Truth']
                # Plot ground truth
                ax.plot(
                    gt_df.index,
                    gt_df['Ground Truth'],
                    label="Ground Truth",
                    color="black",
                    linestyle="--",
                    linewidth=0.8,
                )

                # Aesthetics
                ax.set_title(f" Sample {i} — Offset: {offset}", fontsize=14)
                ax.set_xlabel("Time", fontsize=12, weight="bold")
                ax.set_ylabel("Value", fontsize=12)
                ax.grid(True, linestyle="--", alpha=0.4)
                ax.legend(loc="upper left", fontsize=12)
                plt.xticks(rotation=30)
                plt.tight_layout()
                plt.show()
                plt.savefig(f"plot_sample_{i}_offset_{offset}.png")
                plt.close()

    def _report_plots():
        pass

    def _report_debug_samples():
        pass

    def _persist_artifacts():
        pass
