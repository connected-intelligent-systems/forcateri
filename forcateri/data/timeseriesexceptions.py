class InvalidDataFrameFormat(Exception):
    """Exception raised for errors in the time series data."""

    def __init__(self, message="Invalid time series data provided."):
        self.message = message
        super().__init__(self.message)


class InvalidRepresentationFormat(Exception):
    """Exception raised for errors in the representation format."""

    def __init__(
        self, message="Representation format and provided data are not compatible."
    ):
        self.message = message
        super().__init__(self.message)
