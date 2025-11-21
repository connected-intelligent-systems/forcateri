
from argparse import ArgumentParser
from ast import literal_eval
from pathlib import Path
from inspect import Parameter, signature
import logging

clog = logging.getLogger(__name__)
clover_parser = ArgumentParser()


def _try_eval_literal(s: str, warn_arg_name: str = None):
    warn_arg_name = warn_arg_name or s
    try:
        return literal_eval(s)
    except ValueError as ve:
        if str(ve).startswith("malformed node or string"):
            clog.warning(
                f"Faild to infer python type for {warn_arg_name}; "
                "Assuming type string."
            )
            return s
        else:
            raise ve


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
            clover_parser.add_argument(f"--{fn.__qualname__}.{pn}")
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
            if (p.annotation != Parameter.empty and p.annotation != str) or (
                (p.default != Parameter.empty) and (not isinstance(p.default, str))
            ):
                parsed_args[pname] = _try_eval_literal(
                    parsed_args[pname], f"--{fn.__qualname__}.{pname}"
                )
        types = {k: type(v) for k, v in parsed_args.items()}
        clog.debug(f"Types after evaluation: {types}")

        updated_args = dict(zip(param_names, args))
        updated_args.update(kwargs)
        updated_args.update(parsed_args)
        clog.debug(
            f"Forwarding the following (kw)args to wrapped function: {updated_args}"
        )

        return fn(**updated_args)

    return overridden

# def clover(fn):
#     """
#     Command line override
#     """

#     def overridden(*args, **kwargs):
#         print(
#             f"Calling function {fn.__qualname__} in module {fn.__module__} "
#             f"with args={args} and kwargs={kwargs}"
#         )

#         spam = signature(fn).parameters
#         param_names = spam.keys()

#         print(f"Identified param names:\n{list(param_names)}")
#         for pn, pv in spam.items():
#             #print(f"Parameter {pn} has default value {pv.default} of type {type(pv.default)}")
#             if type(pv.default) == list:
#                 clover_parser.add_argument(f"--{fn.__qualname__}.{pn}", nargs='*', type=type(pv.default[0]) if pv.default else str)
#             else:
#                 clover_parser.add_argument(f"--{fn.__qualname__}.{pn}", type=type(pv.default))
#         parsed_args = vars(clover_parser.parse_known_args()[0])
#         print(f"Parsed the following args from cil:\n{parsed_args}")

#         # dropping Nones for now but unclear how robust that is
#         parsed_args = {
#             k.rsplit(".", 1)[-1]: v
#             for k, v in parsed_args.items()
#             if (v is not None)
#             and (k.rsplit(".", 1)[-1] in param_names)
#             and (k.rsplit(".", 1)[0] == fn.__qualname__)
#         }
#         print(f"Sanitized parsed cli kwargs to:\n{parsed_args}")

#         updated_args = dict(zip(param_names, args))
#         updated_args.update(kwargs)
#         updated_args.update(parsed_args)
#         print(f"Forwarding the following (kw)args to wrapped function:\n{updated_args}")

#         return fn(**updated_args)

#     return overridden