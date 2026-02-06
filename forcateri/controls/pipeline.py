from typing import List, Union

from ..model.modeladapter import ModelAdapter
from ..data.dataprovider import DataProvider
from ..reporting.resultreporter import ResultReporter
import logging


class Pipeline:
    """
    Main point of entry for running the pipeline
    Attributes:
        dp (DataProvider): Responsible for loading and providing datasets
        mad (ModelAdapter): Adapter for training and inference of models.
        rep (ResultReporter): Handles reporting of evaluation results.
    """

    def __init__(
        self,
        data_provider: DataProvider,
        model_adapter: Union[ModelAdapter, List[ModelAdapter]],
        reporter: Union[ResultReporter, List[ResultReporter]],
    ):
        self.data_provider = data_provider
        self.mad = model_adapter if isinstance(model_adapter, list) else [model_adapter]
        self.rep = reporter if isinstance(reporter, list) else [reporter]

    def run(self):
        """
        Executes the pipeline:
        1. Train the model(s) using training/validation sets.
        2. Evaluate the model(s) using test data.
        3. Report all metrics.
        """

        train_set = self.data_provider.get_train_set()
        val_set = self.data_provider.get_val_set()
        test_set = self.data_provider.get_test_set()
        logging.info("Starting pipeline execution.")
        logging.info(f"Training set size: {len(train_set)}")
        logging.info(f"Validation set size: {len(val_set)}")
        logging.info(f"Test set size: {len(test_set)}")

        for model in self.mad:
            model.fit(train_set, val_set)

        # Evaluate and report results
        # results = []
        for reporter in self.rep:
            reporter.report_all(test_data=test_set)
            # results.append(res)
        # return results
