from argparse import ArgumentParser
from ast import literal_eval
from inspect import Parameter, signature
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

clog = logging.getLogger(__name__)
clover_parser = ArgumentParser(conflict_handler="resolve")
global_cfg_dct = dict()


def _try_eval_literal(s, warn_arg_name: Optional[str] = None):
    warn_arg_name = warn_arg_name or s
    if isinstance(s, str):
        try:
            return literal_eval(s)
        except ValueError as ve:
            if str(ve).startswith("malformed node or string"):
                clog.warning(
                    f"Faild to infer python type for arg {warn_arg_name} with value {s}; "
                    "Assuming type string."
                )
                return s
            else:
                raise ve
    else:
        clog.debug(
            f"Skipping literal evaluation since {warn_arg_name} has type {type(s)}."
        )
        return s


def connect_config(config_path: str | Path):
    """
    Params from the config file override params from code
    but not those from cli.
    Calling this method multiple times updates the config dict used by clover.
    """
    with open(config_path, "r") as yamfile:
        cfg_dct = yaml.safe_load(yamfile)

    if cfg_dct is None:
        clog.warning("Connected empty config.")
    else:
        global_cfg_dct.update(cfg_dct)
        clog.debug(f"Adding config {cfg_dct} to clover parser.")
        for k, v in cfg_dct.items():
            clover_parser.add_argument(f"--{k}", default=v)


def clover(fn):
    def overridden(*args, **kwargs):
        clog.debug(
            f"Calling function {fn.__qualname__} in module {fn.__module__} "
            f"with args={args} and kwargs={kwargs}"
        )

        spam = signature(fn).parameters
        param_names = spam.keys()
        clog.debug(f"Identified param names: {list(param_names)}")

        for pn in param_names:
            qual_pn = f"{fn.__qualname__}.{pn}"
            default = global_cfg_dct.get(qual_pn, None)
            clover_parser.add_argument(f"--{qual_pn}", default=default, type=str)
        parsed_args = vars(clover_parser.parse_known_args()[0])
        clog.debug(f"Parsed the following args from cil: {parsed_args}")

        # dropping Nones for now but unclear how robust that is
        parsed_args = {
            k.rsplit(".", 1)[-1]: v
            for k, v in parsed_args.items()
            if (v is not None)
            and (k.rsplit(".", 1)[-1] in param_names)
            and (k.rsplit(".", 1)[0] == fn.__qualname__)
        }
        clog.debug(f"Sanitized parsed cli kwargs to: {parsed_args}")

        for pname in parsed_args.keys():
            p = spam[pname]
            qual_pname = f"{fn.__qualname__}.{pname}"
            if (
                (p.annotation != Parameter.empty and p.annotation != str)
                or ((p.default != Parameter.empty) and (not isinstance(p.default, str)))
                or (qual_pname in global_cfg_dct)
            ):
                parsed_args[pname] = _try_eval_literal(
                    parsed_args[pname], f"--{fn.__qualname__}.{pname}"
                )
        types = {k: type(v) for k, v in parsed_args.items()}
        clog.debug(f"Types after evaluation: {types}")

        updated_args = dict(zip(param_names, args))  # args passed at function call
        updated_args.update(kwargs)  # kwargs passed at function call
        updated_args.update(parsed_args)  # optargs from cli with defaults from config
        clog.debug(
            f"Forwarding the following (kw)args to wrapped function: {updated_args}"
        )

        return fn(**updated_args)

    return overridden
