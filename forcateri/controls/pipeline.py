from typing import List, Union

from ..model.modeladapter import ModelAdapter
from ..data.dataprovider import DataProvider
from ..reporting.resultreporter import ResultReporter


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
        dp: DataProvider,
        mad: Union[ModelAdapter, List[ModelAdapter]],
        rep: Union[ResultReporter, List[ResultReporter]],
    ):
        self.dp = dp
        self.mad = mad if isinstance(mad, list) else [mad]
        self.rep = rep if isinstance(rep, list) else [rep]

    def run(self):
        """
        Executes the pipeline:
        1. Train the model(s) using training/validation sets.
        2. Evaluate the model(s) using test data.
        3. Report all metrics.
        """
        # Example training
        for model in self.mad:
            model.fit(self.dp.get_train_set(), self.dp.get_val_set())

        # Evaluate and report results
        for reporter in self.rep:
            reporter.report_all()
