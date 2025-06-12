
class ClearmlDataMixin:
    def __init__(self, dataset_project:str, dataset_name:str, dataset_version:str, **kwargs):
        super().__init__(**kwargs)
        self.dataset_project = dataset_project
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        
    def get_from_clearml():
        pass 
    def update_on_clearml():
        pass