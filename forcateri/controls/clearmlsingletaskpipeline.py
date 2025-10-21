from forcateri.utils.config_utils import extract_config, load_config
from dotenv import load_dotenv
from pathlib import Path
import os
from clearml import Task
from typing import List, Union, Optional
from .pipeline import Pipeline
from ..model.modeladapter import ModelAdapter
from ..data.dataprovider import DataProvider
from ..reporting.resultreporter import ResultReporter


class ClearMLSingleTaskPipeline(Pipeline):

    def __init__(
        self,
        dp: DataProvider,
        model_adapter: Union[ModelAdapter, List[ModelAdapter]],
        reporter: Union[ResultReporter, List[ResultReporter]],
        config_path: Optional[str] = None,
        project_name: Optional[str] = None,
        task_name: Optional[str] = None,
        param_args: Optional[List] = None,
        requirements: str = None,
        docker: Optional[str] = None,
    ):
        # self.task_name = task_name

        super().__init__(dp, model_adapter, reporter)

        
        self.param_args = param_args or []
        self.requirements = requirements
        self.config = {}
        if config_path and os.path.exists(config_path):
            self.config_path = config_path
            self.config = load_config(config_path)
            #upd config based on param_args and project_name and task_name. Param args have priority
            
            self.args = extract_config(self.config)
            #self.args.append(("config", config_name))
            if self.param_args:
                for k, v in self.param_args:
                    #remove existing arg with same key
                    self.args = [arg for arg in self.args if arg[0] != k]
                    self.args.append((k, v))
        else:
            self.args = self.param_args
            
        clearml_cfg = self.config.get("ClearML", {}).get("task", {}) if isinstance(self.config, dict) else {}
        self.project_name = project_name or clearml_cfg.get("project_name")
        self.task_name = task_name or clearml_cfg.get("task_name")
        self.docker = (
            docker
            or clearml_cfg.get("docker")
            or "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04"
        )




        

    def run(self):
        self.task = Task.init(project_name=self.project_name, task_name=self.task_name)
        function_task = self.task.create_function_task(func=super().run)
        function_task.set_base_docker(docker_image=self.docker)
        function_task.set_packages(self.requirements)
        function_task.connect_configuration(self.args)
        Task.enqueue(task=function_task, queue_name="default")


