from typing import List
import matplotlib.pyplot as plt
import seaborn as sns

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
        self, predictions
    ):  # dont forget to remove predictions after testing
        self.__create_plots(predictions)

    def __compute_metrics():
        pass

    def __select_debug_samples():
        pass

    def __report_metrics():
        pass

    def __create_plots(self, predictions):

        for model in self.models:
            predictions = model.predict(self.test_data)  # Uncomment later after test
            predictions_ts_list = model.to_time_series(
                predictions
            )  # List of TimeSeries objects
            for i, (adapter_input, pred_ts) in enumerate(
                zip(self.test_data, predictions_ts_list)
            ):
                gt_ts = adapter_input.target  # Also a TimeSeries object

                offsets = pred_ts.data.index.get_level_values("offset").unique()

                for offset in offsets:
                    pred_df = pred_ts.by_time(offset)
                    gt_df = gt_ts.by_time(horizon=0)
                    gt_df = gt_df.loc[
                        pred_df.index
                    ]  # Align ground truth with prediction timestamps

                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax = pred_df.plot(
                        title=f"Model: {type(model).__name__} Offset: {offset}",
                        ylabel="Value",
                        xlabel="Time",
                        figsize=(12, 6),
                    )
                    ax.plot(
                        pred_df.index,
                        pred_df[(pred_df.columns[0][0], 0.5)],
                        label="Median Prediction",
                        color="blue",
                        linewidth=2,
                    )
                    ax.fill_between(
                        pred_df.index,
                        pred_df[pred_df.columns[0]],
                        pred_df[pred_df.columns[-1]],
                        color="blue",
                        alpha=0.2,
                        label="Confidence Interval (10%-90%)",
                    )
                    ax.plot(
                        gt_df.index,
                        gt_df.values,
                        label="Ground Truth",
                        color="black",
                        linestyle="--",
                        linewidth=1.5,
                    )
                    ax.set_title(f"{type(model).__name__} — Sample {i} — Offset: {offset}", fontsize=14)
                    ax.set_xlabel("Time", fontsize=12)
                    ax.set_ylabel("Value", fontsize=12)
                    ax.grid(True, linestyle="--", alpha=0.5)
                    ax.legend(fontsize=10)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.show()

    def __report_plots():
        pass

    def __report_debug_samples():
        pass

    def __persist_artifacts():
        pass
