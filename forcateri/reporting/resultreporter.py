from typing import List
import logging

import matplotlib.pyplot as plt
import pandas as pd
import pickle


from .metric import Metric
from ..data.adapterinput import AdapterInput
from ..model.modeladapter import ModelAdapter
from ..data.timeseries import TimeSeries

logger = logging.getLogger(__name__)


class ResultReporter:

    def __init__(
        self,
        # test_data: List[AdapterInput],
        models: List[ModelAdapter],
        metrics: List[Metric],
    ):
        self.test_data = None  # to be set when report_all is called
        self.models = models
        self.metrics = metrics
        self.model_predictions = None  # to be filled after predictions
        self.metric_results = None  # to be filled after metric computation

    def report_all(self, test_data: List[AdapterInput]):
        self.test_data = test_data
        # self.metric_results = self._report_metrics()
        logger.info("Reporting all results...")
        # dont forget to remove predictions after testing
        self._make_predictions()
        self.report_metrics()
        self.report_plots()
        # self.report_debug_samples()

    def _compute_metrics(self):
        logger.info("Computing merics...")
        results = {}

        # loop over each metrics
        for met in self.metrics:
            met_results = {}

            for model, prediction_ts_list in self.model_predictions.items():
                model_results = []
                # loop over test data & predictions
                for i, (adapter_input, pred_ts) in enumerate(
                    zip(self.test_data, prediction_ts_list)
                ):
                    logger.debug(
                        f"Computing metrics for model {model.__class__.__name__} on test series {i}."
                    )

                    gt_ts = adapter_input.target
                    # adjust ground truth length to match pred_ts

                    # reading the max horizon from the model series prediction
                    horizon = pred_ts.offsets.max() // pd.Timedelta(1, pred_ts.freq)
                    logger.debug(f"Horizon determined to be: {horizon}")

                    if horizon < 1:
                        raise ValueError(
                            f"Invalid model adapter output. "
                            f"Horizon is expected to be 1 or greater but was {horizon}."
                        )

                    logger.debug("Aligning predictions and ground truth...")
                    gt_shifted = gt_ts.shift_to_repeat_to_multihorizon(horizon=horizon)
                    logger.debug(f"\ngt_shifted:\n{gt_shifted}")
                    common_index = gt_shifted.data.index.intersection(
                        pred_ts.data.index
                    )
                    logger.debug(
                        f"Common index determined to be\n{common_index.to_frame(index=False)}"
                    )
                    gt_shifted.data = gt_shifted.data.loc[common_index]
                    old_pred_len = len(pred_ts)
                    pred_ts.data = pred_ts.data.loc[common_index]
                    dropped_gt_steps = len(gt_ts) - len(gt_shifted)
                    dropped_pred_steps = old_pred_len - len(pred_ts)
                    if (dropped_gt_steps, dropped_pred_steps) != (0, 0):
                        logger.warning(
                            f"Alignment dropped {dropped_gt_steps} time steps ftom the ground truth "
                            f"and {dropped_pred_steps} time steps from the prediction."
                        )
                    else:
                        logger.debug("No time steps were dropped during alignment.")

                    logger.debug(
                        f"Computing metric {met.__class__.__name__} "
                        f"for model {model.__class__.__name__} "
                        f"on test series {i}..."
                    )
                    reduced_df = met(gt_shifted, pred_ts)
                    model_results.append(reduced_df)

                met_results[model.model_name] = model_results

            results[str(met)] = met_results

        return results

    def _plot_metrics(self, metric_results=None):
        logger.info("Plotting metrics results...")
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
                plt.close()

    def _plot_predictions(self):
        logger.info("Plotting model predictions...")
        for model, prediction_ts_list in self.model_predictions.items():
            for i, (adapter_input, pred_ts) in enumerate(
                zip(self.test_data, prediction_ts_list)
            ):
                gt_ts = adapter_input.target
                offsets = pred_ts.data.index.get_level_values("offset").unique()

                print(f"offsets: {offsets}")
                logger.debug(
                    f"Plotting predictions for model {model.__class__.__name__} on test series {i}."
                )
                if pred_ts.representation == TimeSeries.QUANTILE_REP:
                    for offset in offsets:
                        logger.debug(f"Plotting predictions for offset {offset}.")
                        pred_df = pred_ts.by_time(offset).copy()
                        gt_df = gt_ts.by_time(horizon=0).loc[pred_df.index]

                        # Skip if no data
                        if len(pred_df) <= 1:
                            continue

                        # Flatten MultiIndex columns if needed
                        if isinstance(pred_df.columns, pd.MultiIndex):
                            pred_df.columns = pred_df.columns.get_level_values(
                                1
                            ).astype(float)

                        quantiles = sorted(pred_df.columns.astype(float))
                        lower_q = quantiles[0]
                        upper_q = quantiles[-1]
                        median_q = min(quantiles, key=lambda q: abs(q - 0.5))

                        fig, ax = plt.subplots(figsize=(12, 6))

                        # --- Plot lower quantile (dashed line)
                        ax.plot(
                            pred_df.index,
                            pred_df[lower_q],
                            linestyle="--",
                            color="tab:blue",
                            alpha=0.7,
                            linewidth=1.0,
                            label=f"Lower q={lower_q:.2f}",
                        )

                        # --- Plot upper quantile (dashed line)
                        ax.plot(
                            pred_df.index,
                            pred_df[upper_q],
                            linestyle="--",
                            color="tab:blue",
                            alpha=0.7,
                            linewidth=1.0,
                            label=f"Upper q={upper_q:.2f}",
                        )

                        # --- Plot median quantile (solid line)
                        if median_q in pred_df.columns:
                            ax.plot(
                                pred_df.index,
                                pred_df[median_q],
                                color="tab:blue",
                                linewidth=1.5,
                                label=f"Median q={median_q:.2f}",
                            )

                        # --- Plot ground truth (black dashed)
                        gt_df.columns = ["Ground Truth"]
                        ax.plot(
                            gt_df.index,
                            gt_df["Ground Truth"],
                            color="black",
                            linestyle="--",
                            linewidth=1.2,
                            label="Ground Truth",
                        )

                        # --- Aesthetics
                        ax.set_title(
                            f"{model.__class__.__name__} — Test Series {i} — Offset {offset}"
                        )
                        ax.set_xlabel("Time")
                        ax.set_ylabel("Value")
                        ax.grid(True, linestyle="--", alpha=0.4)
                        ax.legend(loc="upper left", fontsize=9)
                        plt.xticks(rotation=30)
                        plt.tight_layout()
                        plt.show()
                        plt.close()

                elif pred_ts.representation == TimeSeries.DETERM_REP:
                    for offset in offsets:
                        pred_df = pred_ts.by_time(offset).copy()
                        logger.debug(f"pred_df:\n{pred_df.head()}")
                        fig, ax = plt.subplots(figsize=(12, 6))
                        gt_df = gt_ts.by_time(horizon=0).loc[pred_df.index]
                        ax.plot(
                            pred_df.index,
                            pred_df.iloc[:, 0],
                            color="tab:blue",
                            linewidth=1.5,
                            label="Prediction",
                        )
                        ax.plot(
                            gt_df.index,
                            gt_df.iloc[:, 0],
                            color="black",
                            linestyle="--",
                            linewidth=1.2,
                            label="Ground Truth",
                        )
                        ax.set_title(
                            f"{model.__class__.__name__} — Test Series {i} — Offset {offset}"
                        )
                        ax.set_xlabel("Time")
                        ax.set_ylabel("Value")
                        ax.grid(True, linestyle="--", alpha=0.4)
                        ax.legend(loc="upper left", fontsize=9)
                        plt.xticks(rotation=30)
                        plt.tight_layout()
                        plt.show()
                        plt.close()

    def report_metrics(self):
        """Reporting metrics"""
        if self.model_predictions is None:
            logger.debug("report metrics is called before predictions made.")
            self._make_predictions()
        self.metric_results = self._compute_metrics()
        # self._plot_metrics(self.metric_results)
        print(self.metric_results)
        # return self.metric_results

    def _make_predictions(self):
        logger.debug("Making predictions...")
        self.model_predictions = {}
        for model in self.models:
            logger.debug(
                f"Applying model {model.__class__.__name__} to the test data..."
            )
            predictions_ts_list = model.predict(self.test_data)
            logger.debug(
                f"Model {model.__class__.__name__} predictions: len of the predictions list: {len(predictions_ts_list)}"
            )
            self.model_predictions[model] = predictions_ts_list

    def report_plots(self):
        """Reporting plots"""
        if self.model_predictions is None:
            self._make_predictions()
        if self.metric_results is None:
            self.metric_results = self._compute_metrics()
        self._plot_metrics(self.metric_results)
        self._plot_predictions()
        # logger.error("Function _report_plots not implemented.")

    def _persist_artifacts(self):
        logger.error("Function _persist_artifacts not implemented.")

    def _select_debug_samples(self):
        logger.error("Function _select_debug_samples not implemented.")

    def report_debug_samples(self):
        logger.error("Function _report_debug_samples not implemented.")
