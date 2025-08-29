from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from clearml import Task

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
        self.metric_results = self._compute_metrics()
        self._create_plots()
        #return self.metric_results
        #TODO UPLOAD THE RESULTS TO CLEARML
        Task.current_task().upload_artifact(name='Report', artifact_object=self.metric_results)
    def _compute_metrics(self):
        results = []

        # loop over each model's predictions
        for model_idx, prediction_ts_list in enumerate(self.model_predictions):
            model_results = {}

            # loop over each metric
            for met in self.metrics:
                met_results = []

                # loop over test data & predictions
                for adapter_input, pred_ts in zip(self.test_data, prediction_ts_list):
                    gt_ts = adapter_input.target

                    # adjust ground truth length to match pred_ts

                    input_chunk = getattr(self.models[model_idx], "input_chunk_length", 1)
                    reduced_df = met(gt_ts[input_chunk:], pred_ts)
                    met_results.append(reduced_df)

                model_results[met.__class__.__name__] = met_results

            results.append(model_results)

        return results
    
    def _select_debug_samples():
        pass

    def _report_metrics():
        pass
    
    def _make_predictions(self):
        self.model_predictions = []
        for model in self.models:
            predictions_ts_list = model.predict(self.test_data) 
            self.model_predictions.append(predictions_ts_list)
        print(self.model_predictions[0][0].data)
    def _create_plots(self):
        for model_idx, prediction_ts_list in enumerate(self.model_predictions):   
            for i, (adapter_input, pred_ts) in enumerate(zip(self.test_data, prediction_ts_list)):
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
                    ax.set_title(f"Model {model_idx} — Sample {i} — Offset: {offset}", fontsize=14)
                    ax.set_xlabel("Time", fontsize=12, weight="bold")
                    ax.set_ylabel("Value", fontsize=12)
                    ax.grid(True, linestyle="--", alpha=0.4)
                    ax.legend(loc="upper left", fontsize=12)
                    plt.xticks(rotation=30)
                    plt.tight_layout()

              
                    safe_offset = str(offset).replace(" ", "_").replace(":", "-")
                    plt.savefig(f"plot_model_{model_idx}_sample_{i}_offset_{safe_offset}.png")
                    #plt.show()
                    plt.close()

    def _report_plots():
        pass

    def _report_debug_samples():
        pass

    def _persist_artifacts():
        pass
