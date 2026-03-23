import logging
from typing import Any, List, Tuple, Union, Dict
from collections import defaultdict
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
        
        models: List[ModelAdapter],
        metrics: List[Metric],
        test_data: List[AdapterInput] = None,
    ):
        self._is_frozen = False

        self.test_data: List[AdapterInput] = test_data if test_data is not None else []
        self.models: List[ModelAdapter] = []
        self.metrics: List[Metric] = []

        self._computed_predictions = None
        self._computed_metrics = None
        self._computed_debug_samples = None

        
        self._prediction_plots = None
        self._metric_plots = None
        self._debug_sample_plots = None
        if models is not None:
            for model in models:
                self.add_model_adapter(model)
        if metrics is not None:
            for metric in metrics:
                self.add_metric(metric)
        
        self._raw_metric_results = None

    @property
    def is_frozen(self) -> bool:
        """
        Indicates whether the metric computation pipeline has been completed.

        This flag is set to True only after `_compute_metrics` has successfully
        finished processing all registered models and test datasets. It is
        used to lock the reporter state, preventing the addition of new 
        models, data, or metrics once results are finalized.

        Returns:
            bool: True if metrics have been computed, False otherwise.
        """
        return self._is_frozen
    
    @property
    def computed_predictions(self) -> Dict[str, List[TimeSeries]]:
        """
        Provides lazy access to model predictions.

        If predictions have not been generated yet, this property triggers
        `_make_predictions()`, which runs inference for all registered models 
        across all test datasets. Subsequent accesses return the cached results.

        Returns:
            Dict[str, List[TimeSeries]]: A dictionary mapping model names to 
                lists of TimeSeries objects (one per test data entry).

        Note:
            Accessing this property for the first time is a heavy operation 
            proportional to the number of models and the size of the test data.
        """
        if self._computed_predictions is None:
            self.compute_predictions()
        return self._computed_predictions

    @property
    def computed_metrics(self) -> Dict[str, Dict[str, List[pd.DataFrame]]]:
        """
        Provides lazy access to computed metric results.

        Accessing this property triggers the computation of metrics across all 
        registered models and test datasets. If model predictions have not 
        yet been generated, this will automatically trigger `self.predictions` 
        first.

        Returns:
            Dict[str, Dict[str, List[pd.DataFrame]]]: A nested dictionary where:
                - The outer key is the string representation of the Metric.
                - The inner key is the Model name.
                - The value is a list of DataFrames (one per test series).

        Note:
            The results are cached in `self._computed_metrics` after the first 
            computation to avoid redundant processing.
        """
        if self._computed_metrics is None:
            self.compute_metrics()
        return self._computed_metrics
    
    @property
    def computed_debug_samples(self) -> Dict[str,Any]:
        """
        Provides lazy access to computed debug samples.

        Accessing this property triggers the computation of debug samples across all 
        registered models and test datasets. If model predictions have not 
        yet been generated, this will automatically trigger `self.predictions` 
        first.

        Returns:
            Dict[str, Any]: A dictionary mapping debug sample names to their corresponding data.

        Note:
            The results are cached in `self._computed_debug_samples` after the first 
            computation to avoid redundant processing.
        """
        if self._computed_debug_samples is None:
            self.compute_debug_samples()
        return self._computed_debug_samples
    
    @property
    def prediction_plots(self) -> List[Any]:
        """
        Provides access to generated prediction plots.

        Returns:
            List[Any]: A list of figures

        Note:
            This property assumes that `plot_predictions()` has been called 
            beforehand to generate and store the figures in `self._prediction_plots`.
        """
        return self._prediction_plots
    
    @property
    def metric_plots(self) -> List[Any]:
        """
        Provides access to generated metric plots.

        Returns:
            List[Any]: A list of figures

        Note:
            This property assumes that `plot_metrics()` has been called 
            beforehand to generate and store the figures in `self._metric_plots`.
        """
        return self._metric_plots
    
    @property
    def debug_sample_plots(self) -> List[Any]:
        """
        Provides access to generated debug sample plots.

        Returns:
            List[Any]: A list of figures

        Note:
            This property assumes that `plot_debug_samples()` has been called 
            beforehand to generate and store the figures in `self._debug_sample_plots`.
        """
        return self._debug_sample_plots

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
        if self.is_frozen:
            raise RuntimeError(
                "Cannot add new test data after computations have been done. "
                "Please add all test data before calling report_all or report_metrics."
            )
        
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
        if self.is_frozen:
            raise RuntimeError(
                "Cannot add new model after computations have been done. "
                "Please add all models before calling report_all or report_metrics."
            )
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
        if self.is_frozen:
            raise RuntimeError(
                "Cannot add new metric after computations have been done. "
                "Please add all metrics before calling report_all or report_metrics."
            )
        self.metrics.append(metric)

    def report_all(self):
        
        logger.info("Reporting all results...")
        self.compute_predictions()
        self.report_predictions()

        self.compute_metrics()
        self.report_metrics()
        
        self.plot_predictions()
        self.plot_metrics()
        self.compute_debug_samples()
        self.report_debug_samples()

    def compute_all(self):
        logger.info("Computing all results...")
        self.compute_predictions()
        self.compute_metrics()
        self.compute_debug_samples()

    def compute_metrics(self):
        logger.info("Computing merics...")

        def _format_metrics(metric_results) -> List[pd.DataFrame]:
            """
            Computes, aggregates, and formats metric results into digestible DataFrames.

            This method triggers the computation of model predictions and metrics 
            (if not already cached) and then flattens the nested results into 
            consolidated pandas DataFrames, grouped by their column structure.

            The grouping logic ensures that metrics with different output shapes 
            (e.g., scalar metrics vs. vector-based metrics) are returned as 
            separate DataFrames to maintain tabular integrity.

            Returns:
                List[pd.DataFrame]: A list of DataFrames where each DataFrame 
                    contains results for a specific set of metrics. Each DataFrame 
                    includes 'metric', 'model', and 'series_id' as leading columns.

            Note:
                Accessing this method will trigger `self.metric_results`, which in 
                turn triggers `self.predictions` if they have not been computed yet.
                In child classes report results are either saved to disk or uploaded to ClearML, so the returned DataFrames are not necessarily used.
            """         
            
            
            all_results = []

            for metric_name, model_results in metric_results.items():
                for model_name, result_df_list in model_results.items():
                    result = pd.concat(result_df_list, axis=0).copy()
                    result["model"] = model_name
                    result["metric"] = metric_name
                    all_results.append(result)

            
            groups = defaultdict(list)

            for df in all_results:
                # use a tuple of column names as the key
                key = tuple(df.columns)
                groups[key].append(df)

            final_dfs = []

            for dfs in groups.values():
                df_concat = pd.concat(dfs, axis=0, ignore_index=True)
                logger.warning(
                    "Formatted metrics may contain mixed time types (Timestamp and Timedelta), "
                    "depending on the metric and aggregation column."
                )
                
                # reset any index to preserve all data as columns
                df_concat = df_concat.reset_index()
                
                # reorder columns: metric/model/series_id first, everything else after
                id_cols = ["metric", "model", "series_id"]
                other_cols = [c for c in df_concat.columns if c not in id_cols]
                df_concat = df_concat[id_cols + other_cols]
                
                final_dfs.append(df_concat)
            return final_dfs
        
        results = {}

        # loop over each metrics
        for met in self.metrics:
            met_results = {}

            for model_name, prediction_ts_list in self.computed_predictions.items():
                model_results = []
                # loop over test data & predictions
                for i, (adapter_input, pred_ts) in enumerate(
                    zip(self.test_data, prediction_ts_list)
                ):
                    logger.debug(
                        f"Computing metrics for model {model_name} on test series {i}."
                    )

                    gt_ts = adapter_input.target
                    logger.debug(
                        f"Computing metric {str(met)} "
                        f"for model {model_name} "
                        f"on test series {i}..."
                    )
                    reduced_df = met(gt_ts, pred_ts)
                    reduced_df["series_id"] = i
                    reduced_df["model"] = model_name
                    reduced_df["metric"] = met.reduction.__name__
                    model_results.append(reduced_df)

                met_results[model_name] = model_results

            results[str(met)] = met_results
        self._is_frozen = True
        formatted_results = _format_metrics(results)
        self._computed_metrics = formatted_results
        self._raw_metric_results = results

    def report_metrics(self) -> List[pd.DataFrame]:
        logger.info("Reporting metric results...")
        return self.computed_metrics     

    def plot_metrics(self):
        '''
        Note that in child classes returned figure objects are saved or uploaded to clearml
        '''
        logger.info("Plotting metrics results...")

        figures = []
        if self._raw_metric_results is None:
            self.compute_metrics()
        for model_name, model_metrics in self._raw_metric_results.items():
            for metric_name, metric_list in model_metrics.items():
                fig = plot_metric(
                    metric_name=metric_name,
                    metric_list=metric_list,
                    model_name=model_name,
                )

                figures.append((fig, model_name, metric_name))

        self._metric_plots = figures
        

    def plot_predictions(self):
        '''
        Note that in child classes returned figure objects are saved or uploaded to clearml
        '''
        logger.info("Plotting model predictions...")
        figures = []

        for model_name, prediction_ts_list in self.computed_predictions.items():
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

        self._prediction_plots = figures
        
        

    def compute_predictions(self):
        logger.debug("Making predictions...")
        model_predictions = {}
        for model in self.models:
            logger.debug(
                f"Applying model {model.__class__.__name__} to the test data..."
            )
            predictions_ts_list = model.predict(self.test_data)
            logger.debug(
                f"Model {model.__class__.__name__} predictions: len of the predictions list: {len(predictions_ts_list)}"
            )
            model_predictions[model.name] = predictions_ts_list
        self._computed_predictions = model_predictions
        
    
    def report_predictions(self):
        logger.info("Reporting model predictions...")
        return self.computed_predictions


    def report_debug_samples(self):
        logger.warning("Function _report_debug_samples not implemented.")

    def compute_debug_samples(self):
        logger.warning("Function _compute_debug_samples not implemented.")