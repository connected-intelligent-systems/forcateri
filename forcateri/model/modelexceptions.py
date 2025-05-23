class ModelAdapterError(Exception):
    """Base exception for errors in the ModelAdapter."""

    pass


class ModelNotFittedError(ModelAdapterError):
    """Raised when an operation is attempted that requires a fitted model."""

    pass


class InvalidTimeSeriesError(ModelAdapterError):
    """Raised when an invalid TimeSeries object is provided."""

    pass


class InvalidModelTypeError(ModelAdapterError):
    """Raised when invalid model is instanciated"""

    pass
