from forcateri.model.dartsmodels.dartstcnmodel import DartsTCNModel
from forcateri.model.dartsmodels.dartstftmodel import DartsTFTModel
from forcateri.baltbestapi.baltbestaggregatedapidata import BaltBestAggregatedAPIData
import pandas as pd
import yaml
from forcateri.data.dataprovider import DataProvider, SeriesRole
# from darts.models import TCNModel
# from darts.utils.likelihood_models import QuantileRegression
from forcateri.data.timeseries import TimeSeries
from forcateri.reporting.dimwiseaggregatedmetric import DimwiseAggregatedMetric
from forcateri.reporting.dimwiseaggregatedquantileloss import DimwiseAggregatedQuantileLoss
from forcateri.reporting.resultreporter import ResultReporter
from forcateri.controls.pipeline import Pipeline
from pathlib import Path
import argparse
import sys
#from .clearml_entry import extract_config


OFFSET, TIME_STEP = TimeSeries.ROW_INDEX_NAMES
FEATURE, REPRESENTATION = TimeSeries.COL_INDEX_NAMES


def extract_config(config: dict) -> list[tuple]:
    args = []

    for section, section_content in config.items():
        if section == "Models":
            for model_name, params in section_content.items():
                for param_name, param_value in params.items():
                    arg_key = f"model_{model_name}_{param_name}"
                    args.append((arg_key, param_value))
        elif section == "Dataset":
            for dataset_name, dataset_content in section_content.items():
                if isinstance(dataset_content, dict):
                    for subkey, subcontent in dataset_content.items():
                        if subkey == "roles":
                            for role, features in subcontent.items():
                                arg_key = f"Dataset_{dataset_name}_{role}"
                                # If features is a list, join as comma-separated string
                                if isinstance(features, list):
                                    args.append((arg_key, ",".join(features)))
                                else:
                                    args.append((arg_key, features))
                        else:
                            arg_key = f"{dataset_name}_{subkey}"
                            args.append((arg_key, subcontent))
    return args

def from_args_to_kwargs(*args) -> dict:
    kwargs = {"Models": {}, "Dataset": {}}
    for key, value in args:
        if key.startswith("model"):
            keysplit = key.split("_", 2)
            model_name, param = keysplit[1], keysplit[2]
            kwargs["Models"].setdefault(model_name, {})[param] = value
        elif key.startswith("Dataset"):
            _, dataset_name, role_key = key.split("_", 2)
            kwargs["Dataset"].setdefault(dataset_name, {"roles": {}})
            # If value is a comma-separated string, split to list
            if "," in value:
                features = value.split(",")
            else:
                features = [value]
            role_enum = getattr(SeriesRole, role_key.split(".")[-1])
            for f in features:
                kwargs["Dataset"][dataset_name]["roles"][f] = role_enum
    return kwargs

def arg_parser():
    parser = argparse.ArgumentParser()
    project_root=Path(__file__).parent.parent
    config_path = project_root.joinpath("configs")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Configuration file name without .yaml extension"
    )
    args,remaining_args = parser.parse_known_args()
    with open(config_path.joinpath(args.config + '.yaml'),"r") as infile:
            parsed_config = yaml.safe_load(infile)
    args = extract_config(parsed_config)
    for k, v in args:
        if isinstance(v, list):
            v = ",".join(map(str, v))
        elif v is None:
            v = "None"
        parser.add_argument(f"--{k}", default=v)
    return parser

def main(*args):



    ds0 = BaltBestAggregatedAPIData()
    kwargs = from_args_to_kwargs(*args)
    #roles = kwargs['roles']

    # roles = {
    #     'q_hca': SeriesRole.TARGET, 
    #     'temperature_outdoor_avg':SeriesRole.KNOWN, 
    #     'temperature_1_max':SeriesRole.OBSERVED, 
    #     'temperature_2_max':SeriesRole.OBSERVED,
    #     'temperature_room_avg':SeriesRole.OBSERVED,}
    roles = kwargs['Dataset']['Baltbestapi']['roles']
    print(roles)
    
    dp = DataProvider(data_sources=[ds0], roles=roles)
    #print(kwargs['Models'].keys())
    mad0 = DartsTCNModel(kwargs=kwargs['Models']['DartsTCNModel'])
    mad1 = DartsTFTModel(kwargs=kwargs['Models']['DartsTFTModel'])
    met0 = DimwiseAggregatedQuantileLoss(axes=[OFFSET])
    #met0 = DimwiseAggregatedMetric(axes=[OFFSET])
    met1 = DimwiseAggregatedMetric(axes=[TIME_STEP])
    test_set = dp.get_test_set()
    rep = ResultReporter(test_set,[mad0,mad1],[met0,met1])
    #rep.report_all()
    pipe = Pipeline(dp,model_adapter=[mad0,mad1],reporter=rep)
    #results = pipe.run()
    pipe.run()
    #return results

if __name__ == "__main__":
    
    parser = arg_parser()
    args = parser.parse_args()
    print("pipeline args\n\n")
    print(args)

    print("\n\n")
    print(*list(vars(args).items()))

    # Parse the command-line arguments


    main(*list(vars(args).items()))
