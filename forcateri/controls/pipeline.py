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
        model_adapter: Union[ModelAdapter, List[ModelAdapter]],
        reporter: Union[ResultReporter, List[ResultReporter]],
    ):
        self.dp = dp
        self.mad = model_adapter if isinstance(model_adapter, list) else [model_adapter]
        self.rep = reporter if isinstance(reporter, list) else [reporter]

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
        #results = []
        for reporter in self.rep:
            reporter.report_all()
            #results.append(res)
        #return results
