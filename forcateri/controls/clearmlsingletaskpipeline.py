from forcateri.utils.config_utils import extract_config, load_config
from dotenv import load_dotenv
from pathlib import Path
import os
from clearml import Task
from typing import List, Union
from .pipeline import Pipeline
from ..model.modeladapter import ModelAdapter
from ..data.dataprovider import DataProvider
from ..reporting.resultreporter import ResultReporter


class ClearMlSingleTaskPipeline(Pipeline):
    def __init__(
        self,
        config_name: str,
        project_root: Path,
        dp: DataProvider,
        model_adapter: Union[ModelAdapter, List[ModelAdapter]],
        reporter: Union[ResultReporter, List[ResultReporter]],
    ):
        # self.task_name = task_name
        super().__init__(dp, model_adapter, reporter)
        load_dotenv()
        self.config_name = config_name
        self.config = load_config(config_name, project_root)
        self.args = extract_config(self.config)
        self.args.append(("config", config_name))

    def execute_remotely(self):
        token = os.environ["GIT_TOKEN"]
        self.task.execute_remotely(
            queue_name="default",
            clone=False,
            exit_process=True,
            docker=self.config["ClearML"]["task"]["docker"],
            docker_args=(
                f"-e CLEARML_AGENT_GIT_USER=oauth2 -e CLEARML_AGENT_GIT_PASS={token}"
            ),
        )

    def execute_task_enq(self):
        token = os.environ["GIT_TOKEN"]

        self.task = Task.create(
            project_name=self.config["ClearML"]["task"]["project_name"],
            task_name=self.config["ClearML"]["task"]["task_name"],
            add_task_init_call=True,
            branch=self.config["ClearML"]["task"]["branch"],
            repo=self.config["ClearML"]["task"]["repo"],
            script=self.config["ClearML"]["task"]["script"],
            # docker = "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
            docker=self.config["ClearML"]["task"]["docker"],
            docker_args=(
                f"-e CLEARML_AGENT_GIT_USER=oauth2 -e CLEARML_AGENT_GIT_PASS={token} -e CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL={self.config['ClearML']['task']['skip_env_install']}"
            ),
            argparse_args=self.args,
        )
        Task.enqueue(task=self.task, queue_name="default")
