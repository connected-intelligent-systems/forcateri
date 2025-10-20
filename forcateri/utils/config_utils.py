from forcateri.data.dataprovider import SeriesRole
from forcateri.data.timeseries import TimeSeries
#from forcateri import project_root
import argparse
import yaml
from pathlib import Path

def extract_config(config: dict) -> list[tuple]:
    args = []
    for section, section_content in config.items():
        if section == "ClearML":
            continue
        elif section == "Models":
            for model_name, params in section_content.items():
                for param_name, param_value in params.items():
                    arg_key = f"model.{model_name}.{param_name}"
                    args.append((arg_key, param_value))
        elif section == "DataProvider":
            for param_name, param_value in section_content.items():
                arg_key = f"DataProvider.{param_name}"
                args.append((arg_key, param_value))
        elif section == "DataSources":
            for dataset_name, dataset_content in section_content.items():
                if isinstance(dataset_content, dict):
                    for subkey, subcontent in dataset_content.items():
                        if subkey == "roles":
                            for role, features in subcontent.items():
                                arg_key = f"DataSources.{dataset_name}.{role}"
                                args.append((arg_key, features))
                        else:
                            arg_key = f"{dataset_name}.{subkey}"
                            args.append((arg_key, subcontent))
        elif section == "Metrics":
            for metric_name, params in section_content.items():
                for param_name, param_value in params.items():
                    arg_key = f"Metric.{metric_name}.{param_name}"
                    args.append((arg_key, param_value))

    return args

def from_args_to_kwargs(*args) -> dict:
    """Simple version - no string-to-list conversion needed since we keep lists as lists"""
    kwargs = {"Models": {}, "DataSources": {}, "DataProvider": {}, "Metrics": {}}
    for key, value in args:
        if key.startswith("model"):
            keysplit = key.split(".", 2)
            model_name, param = keysplit[1], keysplit[2]
            kwargs["Models"].setdefault(model_name, {})[param] = value
        elif key.startswith("DataSources"):
            _, dataset_name, role_key = key.split(".", 2)
            kwargs["DataSources"].setdefault(dataset_name, {"roles": {}})
            # Handle both single values and lists
            features = value if isinstance(value, list) else [value]
            role_enum = getattr(SeriesRole, role_key)
            for f in features:
                kwargs["DataSources"][dataset_name]["roles"][f] = role_enum
        elif key.startswith("DataProvider"):
            _, param = key.split(".", 1)
            kwargs["DataProvider"][param] = value
        elif key.startswith("Metric"):
            keysplit = key.split(".", 2)
            metric_name, param = keysplit[1], keysplit[2]
            # Ensure axes is always a list
            if param == "axes" and not isinstance(value, list):
                value = [value]
            kwargs["Metrics"].setdefault(metric_name, {})[param] = value
    return kwargs

def arg_parser(project_root):
    parser = argparse.ArgumentParser()
    config_path = project_root.joinpath("configs")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Configuration file name without .yaml extension"
    )
    args, remaining_args = parser.parse_known_args()
    with open(config_path.joinpath(args.config + '.yaml'), "r") as infile:
        parsed_config = yaml.safe_load(infile)
    args = extract_config(parsed_config)
    for k, v in args:
        if isinstance(v, list):
            # Keep lists as lists using nargs='*'
            parser.add_argument(f"--{k}", default=v, nargs='*', type=type(v[0]) if v else str)
        elif v is None:
            parser.add_argument(f"--{k}", default=None)
        else:
            parser.add_argument(f"--{k}", default=v, type=type(v))
    return parser

def load_config(config_path: Path) -> dict:
    """
    Parse command line args and load YAML config file from configs/ directory.
    Returns: (config_name: str, parsed_config: dict)
    """
    parser = argparse.ArgumentParser(description="Pipeline config parser")
    parser.add_argument(
        '--config',
        type=str,
        default='pipeline',
        help='Specify the config name (without .yaml) from configs/ directory'
    )
    args = parser.parse_args()

    #project_root = Path(__file__).parent.parent
    #config_path = project_root / "configs" / f"{config_name}.yaml"

    with open(config_path, "r") as infile:
        parsed_config = yaml.safe_load(infile)

    return parsed_config