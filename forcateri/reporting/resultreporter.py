import logging
from typing import List

import pandas as pd
import plotly.graph_objects as go

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

            for model_name, prediction_ts_list in self.model_predictions.items():
                model_results = []
                # loop over test data & predictions
                for i, (adapter_input, pred_ts) in enumerate(
                    zip(self.test_data, prediction_ts_list)
                ):
                    logger.debug(
                        f"Computing metrics for model {model_name.__class__.__name__} on test series {i}."
                    )

                    gt_ts = adapter_input.target
                    logger.debug(
                        f"Computing metric {met.__class__.__name__} "
                        f"for model {model_name.__class__.__name__} "
                        f"on test series {i}..."
                    )
                    reduced_df = met(gt_ts, pred_ts)
                    model_results.append(reduced_df)

                met_results[model_name] = model_results

            results[str(met)] = met_results

        return results

    def _plot_metrics(self, metric_results=None):
        logger.info("Plotting metrics results...")
        if metric_results is None:
            metric_results = self.metric_results

        figures = []

        for model_name, model_metrics in metric_results.items():
            for metric_name, metric_list in model_metrics.items():
                fig = go.Figure()
                xlabel = "Index"

                for i, df in enumerate(metric_list):
                    if not isinstance(df, pd.DataFrame) or len(df) <= 1:
                        continue

                    # Determine x-axis
                    if isinstance(df.index, pd.MultiIndex):
                        level = 1 if len(df.index.names) > 1 else 0
                        x = df.index.get_level_values(level)
                        xlabel = df.index.names[level]
                    else:
                        x = df.index

                    # Add traces
                    for col in df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=df[col],
                                mode="lines",
                                name=f"Test series id: {i} - {col}",
                            )
                        )

                # Set layout
                fig.update_layout(
                    title=f"{metric_name} for {model_name}",
                    xaxis_title=xlabel,
                    yaxis_title="Metric Value",
                    legend_title="Series",
                    template="plotly_white",
                    autosize=True,
                )

                figures.append((fig, model_name, metric_name))

        return figures

    def _plot_predictions(self):
        logger.info("Plotting model predictions...")
        figures = []

        for model, prediction_ts_list in self.model_predictions.items():
            for i, (adapter_input, pred_ts) in enumerate(
                zip(self.test_data, prediction_ts_list)
            ):
                gt_ts = adapter_input.target
                offsets = pred_ts.data.index.get_level_values("offset").unique()

                logger.debug(
                    f"Plotting predictions for model {model.__class__.__name__} on test series {i}."
                )

                if pred_ts.representation == TimeSeries.QUANTILE_REP:
                    for offset in offsets:
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
                        lower_q, upper_q = quantiles[0], quantiles[-1]
                        median_q = min(quantiles, key=lambda q: abs(q - 0.5))

                        fig = go.Figure()

                        # Lower quantile
                        fig.add_trace(
                            go.Scatter(
                                x=pred_df.index,
                                y=pred_df[lower_q],
                                mode="lines",
                                name=f"Lower q={lower_q:.2f}",
                                line=dict(dash="dash", color="blue", width=1),
                                opacity=0.7,
                            )
                        )

                        # Upper quantile
                        fig.add_trace(
                            go.Scatter(
                                x=pred_df.index,
                                y=pred_df[upper_q],
                                mode="lines",
                                name=f"Upper q={upper_q:.2f}",
                                line=dict(dash="dash", color="blue", width=1),
                                opacity=0.7,
                            )
                        )

                        # Median quantile
                        if median_q in pred_df.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=pred_df.index,
                                    y=pred_df[median_q],
                                    mode="lines",
                                    name=f"Median q={median_q:.2f}",
                                    line=dict(color="blue", width=2),
                                )
                            )

                        # Ground truth
                        gt_df.columns = ["Ground Truth"]
                        fig.add_trace(
                            go.Scatter(
                                x=gt_df.index,
                                y=gt_df["Ground Truth"],
                                mode="lines",
                                name="Ground Truth",
                                line=dict(color="black", dash="dash", width=2),
                            )
                        )

                        # Layout
                        fig.update_layout(
                            title=f"{model} — Test Series {i} — Offset {offset}",
                            xaxis_title="Time",
                            yaxis_title="Value",
                            legend_title="Series",
                            template="plotly_white",
                        )

                        figures.append((fig, model, i, offset))

                elif pred_ts.representation == TimeSeries.DETERM_REP:
                    for offset in offsets:
                        pred_df = pred_ts.by_time(offset).copy()
                        gt_df = gt_ts.by_time(horizon=0).loc[pred_df.index]

                        fig = go.Figure()
                        fig.add_trace(
                            go.Scatter(
                                x=pred_df.index,
                                y=pred_df.iloc[:, 0],
                                mode="lines",
                                name="Prediction",
                                line=dict(color="blue", width=2),
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=gt_df.index,
                                y=gt_df.iloc[:, 0],
                                mode="lines",
                                name="Ground Truth",
                                line=dict(color="black", dash="dash", width=2),
                            )
                        )

                        fig.update_layout(
                            title=f"{model} — Test Series {i} — Offset {offset}",
                            xaxis_title="Time",
                            yaxis_title="Value",
                            legend_title="Series",
                            template="plotly_white",
                        )

                        figures.append((fig, model, i, offset))

        return figures

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
            self.model_predictions[model.name] = predictions_ts_list

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
