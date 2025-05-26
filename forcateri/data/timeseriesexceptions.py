
class InvalidDataFrameFormat(Exception):
    """Exception raised for errors in the time series data."""
    def __init__(self, message="Invalid time series data provided."):
        self.message = message
        super().__init__(self.message)