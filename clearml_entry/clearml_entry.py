from clearml import Task
from .test_pipeline import main

my_task = Task.init(
    project_name="ForeSightNEXT/BaltBest",
    task_name="Forcateri Pipeline Test",
)

if __name__ == "__main__":
    
    print("test local")
    my_task.execute_remotely(
        queue_name="default",
        clone=False,
        exit_process=True,
    )
    print("Test on remote")
    results = main()
    Task.current_task().upload_artifact(name='results', artifact_object=results)