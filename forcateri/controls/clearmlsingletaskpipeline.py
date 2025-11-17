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
        init_args: Optional[List] = None,
        requirements: str = None,
        docker: Optional[str] = None,
        repo: Optional[str] = None,
        branch: Optional[str] = None,
    ):
        # self.task_name = task_name

        super().__init__(dp, model_adapter, reporter)


        self.init_args = init_args or []
        self.requirements = requirements
        self.config = {}
        if config_path and os.path.exists(config_path):
            self.config_path = config_path
            self.config = load_config(config_path)
            #upd config based on param_args and project_name and task_name. Param args have priority
            
            self.parsed_cfg_args = extract_config(self.config)
            #self.args.append(("config", config_name))
            #update parsed_cfg_args with init_args
            if self.init_args:
                for k, v in self.init_args:
                    #remove existing arg with same key
                    
                    self.parsed_cfg_args = [arg for arg in self.parsed_cfg_args if arg[0] != k]
                    self.parsed_cfg_args.append((k, v))
        else:
            self.parsed_cfg_args = self.init_args

        clearml_cfg = self.config.get("ClearML", {}).get("task", {}) if isinstance(self.config, dict) else {}
        self.project_name = project_name or clearml_cfg.get("project_name")
        self.task_name = task_name or clearml_cfg.get("task_name")
        self.docker = (
            docker
            or clearml_cfg.get("docker")
        )
        self.repo = repo or clearml_cfg.get("repo")
        self.branch = branch or clearml_cfg.get("branch")



        

    def run(self):
        self.task = Task.init(project_name=self.project_name, task_name=self.task_name)
        function_task = self.task.create_function_task(func=super().run)
        function_task.set_base_docker(docker_image=self.docker)
        function_task.set_packages(self.requirements)
        #function_task.connect_configuration(self.parsed_cfg_args, ignore_remote_overrides= False)
        function_task.set_repo(repo=self.repo, branch=self.branch)
        Task.enqueue(task=function_task, queue_name="default")


