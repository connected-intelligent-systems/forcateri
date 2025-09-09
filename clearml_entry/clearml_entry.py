from clearml import Task
import os
from dotenv import load_dotenv
from .test_pipeline import main
from pathlib import Path
import argparse
import yaml

load_dotenv()
def exec_remotely():

    my_task = Task.init(
        project_name="ForeSightNEXT/BaltBest",
        task_name="Forcateri Pipeline Test",
    )
    print("test local")
    my_task.execute_remotely(
        queue_name="default",
        clone=False,
        exit_process=True,
    )
    print("Test on remote")
    results = main()
    Task.current_task().upload_artifact(name='results', artifact_object=results)
def exec_taskenq(**kwargs):
    token = os.environ["GIT_OAUTH_TOKEN"]
    task = Task.create(
        project_name="ForeSightNEXT/BaltBest",
        task_name="test",
        add_task_init_call=True,
        branch="main",
        repo="git@github.com:connected-intelligent-systems/forcateri.git",
        script="clearml_entry/test_pipeline.py",
        docker = "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04",
        docker_args=(
            f"-e CLEARML_AGENT_GIT_USER=oauth2 -e CLEARML_AGENT_GIT_PASS={token}"
        ),
        argparse_args=list(kwargs.items()),
    )
    Task.enqueue(task=task, queue_name="default")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline config parser")
    parser.add_argument(
        '--config',
        type = str,
        #required = True,
        default = 'tcn_pipeline',
        help = 'Specify the config for the training process'
    )
    args = parser.parse_args()
    project_root=Path(__file__).parent.parent
    config_path = project_root.joinpath("configs")

    with open(config_path.joinpath(args.config + '.yaml'),"r") as infile:
            parsed_config = yaml.safe_load(infile)
    print(parsed_config.keys())
    #exec_taskenq(**parsed_config)
    
    