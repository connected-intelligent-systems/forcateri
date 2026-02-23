import logging
from typing import List, Union

import pandas as pd
import plotly.graph_objects as go

from .metric import Metric
from ..data.adapterinput import AdapterInput
from ..model.modeladapter import ModelAdapter
from ..data.timeseries import TimeSeries
from .plotting import plot_metric, plot_quantile_predictions, plot_determ_predictions

logger = logging.getLogger(__name__)


class ResultReporter:
    """
    Handles model predictions, metric computation, and plotting.

    Test data, models, and metrics can be registered either at initialization
    or incrementally using `add_test_data`, `add_model_adapter`, and `add_metric`.

    Constructor Arguments:
        test_data (AdapterInput or list of AdapterInput, optional):
            Test datasets to register immediately.
        models (list of ModelAdapter, optional):
            Models to register immediately.
        metrics (list of Metric, optional):
            Metrics to register immediately.

    Note:
        Using the constructor arguments is convenient for simple initialization
        when all entities are already available. Calling the `add_*` methods
        separately can be useful for lazy loading or dynamically adding new
        test data, models, or metrics after the reporter has been created.
    """

    def __init__(
        self,
        # test_data: List[AdapterInput],
        models: List[ModelAdapter],
        metrics: List[Metric],
    ):
        self.test_data: List[AdapterInput] = []  # to be set when report_all is called
        self.models: List[ModelAdapter] = []
        self.metrics: List[Metric] = []
        self.model_predictions = None
        self.metric_results = None

        if models is not None:
            for model in models:
                self.add_model_adapter(model)
        if metrics is not None:
            for metric in metrics:
                self.add_metric(metric)

    def report_all(self, test_data: List[AdapterInput]):
        ResultReporter.add_test_data(test_data)
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
                fig = plot_metric(
                    metric_name=metric_name,
                    metric_list=metric_list,
                    model_name=model_name,
                )

                figures.append((fig, model_name, metric_name))

        return figures

    def _plot_predictions(self):
        logger.info("Plotting model predictions...")
        figures = []

        for model_name, prediction_ts_list in self.model_predictions.items():
            for id, (adapter_input, pred_ts) in enumerate(
                zip(self.test_data, prediction_ts_list)
            ):
                gt_ts = adapter_input.target
                offsets = pred_ts.data.index.get_level_values("offset").unique()

                logger.debug(
                    f"Plotting predictions for model {model_name} on test series {id}."
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
                        fig = plot_quantile_predictions(
                            quantiles=quantiles,
                            pred_df=pred_df,
                            gt_df=gt_df,
                            offset=offset,
                            model_name=model_name,
                            test_series_id=id,
                        )

                        figures.append((fig, model_name, id, offset))

                elif pred_ts.representation == TimeSeries.DETERM_REP:
                    for offset in offsets:
                        pred_df = pred_ts.by_time(offset).copy()
                        gt_df = gt_ts.by_time(horizon=0).loc[pred_df.index]
                        fig = plot_determ_predictions(
                            pred_df=pred_df,
                            gt_df=gt_df,
                            offset=offset,
                            model_name=model_name,
                            test_series_id=id,
                        )

                        figures.append((fig, model_name, id, offset))

        return figures

    def add_test_data(self, test_data: Union[AdapterInput, List[AdapterInput]]):
        """
        Register one or more test datasets to the reporter.

        Args:
            test_data (AdapterInput or list of AdapterInput):
                A single test dataset or a list of datasets to add.

        Note:
            Useful for incrementally adding test data after initialization,
            e.g., when loading data lazily or in multiple batches.
        """
        if isinstance(test_data, AdapterInput):
            self.test_data.append(test_data)
        else:
            self.test_data.extend(test_data)

    def add_model_adapter(self, model_adapter: ModelAdapter):
        """
        Register a model adapter to the reporter.

        Args:
            model_adapter (ModelAdapter): The model to add.

        Note:
            Useful for incrementally adding models after initialization,
            for example when models are created or loaded dynamically.
        """
        self.models.append(model_adapter)

    def add_metric(self, metric: Metric):
        """
        Register a metric to the reporter.

        Args:
            metric (Metric): The metric to add.

        Note:
            Useful for incrementally adding metrics after initialization,
            e.g., when metrics are defined or loaded later.
        """
        self.metrics.append(metric)

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
