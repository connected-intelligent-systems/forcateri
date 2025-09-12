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
#from .clearml_entry import extract_config


OFFSET, TIME_STEP = TimeSeries.ROW_INDEX_NAMES
FEATURE, REPRESENTATION = TimeSeries.COL_INDEX_NAMES

def parse_dynamic_args():
    parser = argparse.ArgumentParser(add_help=False)
    _, unknown = parser.parse_known_args()
    it = iter(unknown)
    pairs = []
    for tok in it:
        if tok.startswith("--"):
            key = tok[2:]
            val = next(it, None)
            if val is None or val.startswith("--"):
                pairs.append((key, "True"))
                if val and val.startswith("--"):
                    it = iter([val] + list(it))
            else:
                # restore lists
                if "," in val and not any(c in val for c in "{}[]"):
                    parts = val.split(",")
                    try_cast = []
                    for p in parts:
                        if p == "None":
                            try_cast.append(None)
                        else:
                            try:
                                try_cast.append(int(p))
                            except:
                                try:
                                    try_cast.append(float(p))
                                except:
                                    try_cast.append(p)
                    val = try_cast
                elif val == "None":
                    val = None
                pairs.append((key, val))
    return pairs

def from_args_to_kwargs(*args) -> dict:
    """
    Convert list of tuples (from extract_config) back into structured kwargs.
    Example input:
      ('DartsTFTModel_input_chunk_length', 7)
      ('DartsTCNModel_kernel_size', 3)
      ('Baltbestapi_TARGET', 'q_hca')
    """
    kwargs = {"Models": {},  "Dataset": {}}
    for key, value in args:
        #print(key)
        if key.startswith("model"):  # Model config
            keysplit = key.split("_",2)
            model_name, param = keysplit[1], keysplit[2]
            
            kwargs["Models"].setdefault(model_name, {})[param] = value

        elif key.startswith("Dataset"):  # Dataset or role
            _, dataset_name, subkey = key.split("_", 2)
            print(subkey)
            kwargs['Dataset'].setdefault(dataset_name,{"roles":{}})
            if subkey in ["SeriesRole.TARGET", "SeriesRole.KNOWN", "SeriesRole.OBSERVED"]:
                kwargs["Dataset"][dataset_name]["roles"][value] = subkey

    return kwargs

def main(*args):

    if not args:
        # Running under ClearML: pull from sys.argv
        raw_pairs = parse_dynamic_args()
    else:
        # Direct python call with tuples passed in
        raw_pairs = args
    kwargs = from_args_to_kwargs(*raw_pairs)

    ds0 = BaltBestAggregatedAPIData()

    #roles = kwargs['roles']

    # roles = {
    #     'q_hca': SeriesRole.TARGET, 
    #     'temperature_outdoor_avg':SeriesRole.KNOWN, 
    #     'temperature_1_max':SeriesRole.OBSERVED, 
    #     'temperature_2_max':SeriesRole.OBSERVED,
    #     'temperature_room_avg':SeriesRole.OBSERVED,}
    roles = kwargs['Dataset']['Baltbestapi']['roles']
    roles_enum = {
        feature: getattr(SeriesRole, role_str.split('.')[-1])
        for feature, role_str in roles.items()
    }
    
    dp = DataProvider(data_sources=[ds0], roles=roles_enum)
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
    
    # project_root=Path(__file__).parent.parent
    # config_path = project_root.joinpath("configs")
    # config_name = 'tft_pipeline'
    # with open(config_path.joinpath(config_name + '.yaml'),"r") as infile:
    #         parsed_config = yaml.safe_load(infile)
    # args = extract_config(parsed_config)
    
    main()
