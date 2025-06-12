from clearml import Dataset, Task


class ClearmlDataMixin:
    def __init__(
        self, dataset_project: str, dataset_name: str, dataset_version: str, **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset_project = dataset_project
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version

    def get_from_clearml(self):
        data_root = Dataset.get(
            dataset_project=self.dataset_project,
            dataset_name=self.dataset_name,
            dataset_version=self.dataset_version,
        )
        return data_root

    def update_on_clearml():
        pass
