import inspect
from copy import deepcopy

class DefaultParamsMixin:
    @classmethod
    def get_default_params(cls):
        """Return all default parameters from the class's __init__ method (signature + kwargs.get)."""
        defaults = {}
        init_method = cls.__init__

        # Extract defaults from __init__ function
        sig = inspect.signature(init_method)
        for name, param in sig.parameters.items():
            # param.default is 'inspect._empty' if there’s no default
            if param.default is not inspect._empty:
                defaults[name] = param.default

        # Extract defaults from kwargs.get() in the body
        try:
            source = inspect.getsource(init_method)
        except OSError:
            source = ""

        for line in source.splitlines():
            line = line.strip()
            if 'kwargs.get' in line:
                try:
                    key = line.split('kwargs.get(')[1].split(',')[0].strip().strip('"\'')
                    default_str = line.split(',', 1)[1].rsplit(')', 1)[0].strip()
                    defaults[key] = eval(default_str)
                except Exception:
                    pass

        return deepcopy(defaults)
    
    def get_default_args(self):
        args = []
        for k,v in self.get_default_params().items():
            args.append((k,v))
        return args
