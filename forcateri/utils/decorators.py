from argparse import ArgumentParser

from inspect import signature

clover_parser = ArgumentParser()

def clover(fn):
    """
    Command line override
    """

    def overridden(*args, **kwargs):
        print(
            f"Calling function {fn.__qualname__} in module {fn.__module__} "
            f"with args={args} and kwargs={kwargs}"
        )

        spam = signature(fn).parameters
        param_names = spam.keys()

        print(f"Identified param names:\n{list(param_names)}")
        for pn, pv in spam.items():
            #print(f"Parameter {pn} has default value {pv.default} of type {type(pv.default)}")
            if type(pv.default) == list:
                clover_parser.add_argument(f"--{fn.__qualname__}.{pn}", nargs='*', type=type(pv.default[0]) if pv.default else str)
            else:
                clover_parser.add_argument(f"--{fn.__qualname__}.{pn}", type=type(pv.default))
        parsed_args = vars(clover_parser.parse_known_args()[0])
        print(f"Parsed the following args from cil:\n{parsed_args}")

        # dropping Nones for now but unclear how robust that is
        parsed_args = {
            k.rsplit(".", 1)[-1]: v
            for k, v in parsed_args.items()
            if (v is not None)
            and (k.rsplit(".", 1)[-1] in param_names)
            and (k.rsplit(".", 1)[0] == fn.__qualname__)
        }
        print(f"Sanitized parsed cli kwargs to:\n{parsed_args}")

        updated_args = dict(zip(param_names, args))
        updated_args.update(kwargs)
        updated_args.update(parsed_args)
        print(f"Forwarding the following (kw)args to wrapped function:\n{updated_args}")

        return fn(**updated_args)

    return overridden