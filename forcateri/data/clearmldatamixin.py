from clearml import Dataset, Task
from pathlib import Path

class ClearmlDataMixin:

    def get_from_clearml(self):
        data_root = Dataset.get(
            dataset_project=self.dataset_project,
            dataset_name=self.dataset_name,
            #dataset_version=self.dataset_version,
        ).get_local_copy()
        data_root = Path(data_root)
        data_root = data_root / self.filename
        return data_root
    def link_dataset(self, dataset_project: str, dataset_name: str, file_name: str):
        self.dataset_project = dataset_project
        self.dataset_name = dataset_name
        self.filename = file_name
        #self.dataset_version = dataset_version
    def update_on_clearml():
        pass
