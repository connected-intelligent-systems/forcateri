import pandas as pd

from forcateri import TimeSeries

from .common_fixtures import *


class TestTimeSeriesInitialization:
    @staticmethod
    def assert_basic_attributes(ts, expected_rep, expected_quant):
        """
        All of the core fields should be set according to the consructor arguments
        and exist in every case (even if they are None)
        """
        assert isinstance(ts, TimeSeries)
        assert hasattr(ts, "representation")
        assert hasattr(ts, "quantiles")
        assert hasattr(ts, "data")
        assert ts.representation == expected_rep
        assert ts.quantiles == expected_quant

    @staticmethod
    def assert_inner_compatibility(ts, representation):
        """
        The data frame assigned to a TimeSerie's data attribute
        should be of both matching and compatible format
        """
        assert TimeSeries.is_matching_format(ts.data, representation)
        assert TimeSeries.is_compatible_format(ts.data, representation)

    # initialization with matching format

    def test_matching_determ(self, determ_df):
        rep = TimeSeries.DETERM_REP
        ts = TimeSeries(determ_df, rep)
        self.assert_basic_attributes(ts, rep, None)
        self.assert_inner_compatibility(ts, rep)
        pd.testing.assert_frame_equal(determ_df, ts.data)

    def test_matching_determ_inference(self, determ_df):
        """
        If neither a representation argument nor a quantiles argument is passed,
        the representation should be assumed to be TimeSeries.DETERM_REP.
        """
        rep = TimeSeries.DETERM_REP
        ts = TimeSeries(determ_df)
        self.assert_basic_attributes(ts, rep, None)
        self.assert_inner_compatibility(ts, rep)
        pd.testing.assert_frame_equal(determ_df, ts.data)

    def test_matching_quantile(self, quantile_df, quantiles):
        rep = TimeSeries.QUANTILE_REP
        ts = TimeSeries(quantile_df, rep, quantiles)
        self.assert_basic_attributes(ts, rep, quantiles)
        self.assert_inner_compatibility(ts, rep)

        # Series in quantile representation have a quantiles parameter
        assert ts.quantiles is not None

        # ts.quantiles is of a type that supports len()
        assert hasattr(ts.quantiles, "__len__")

        # as many entries in self.quantiles as implied by feature representation
        assert len(ts.quantiles) == len(ts.data.columns.get_level_values(1).unique())
        pd.testing.assert_frame_equal(quantile_df, ts.data)

    def test_matching_quantile_inference(self, quantile_df, quantiles):
        """
        If a representation argument is not passed but a quantiles argument is,
        the representation should be assumed to be TimeSeries.QUANTILE_REP.
        """
        rep = TimeSeries.QUANTILE_REP
        ts = TimeSeries(quantile_df, quantiles=quantiles)
        self.assert_basic_attributes(ts, rep, quantiles)
        self.assert_inner_compatibility(ts, rep)
        pd.testing.assert_frame_equal(quantile_df, ts.data)

    def test_matching_sample(self, sample_df):
        rep = TimeSeries.SAMPLE_REP
        ts = TimeSeries(sample_df, rep)
        self.assert_basic_attributes(ts, rep, None)
        self.assert_inner_compatibility(ts, rep)
        pd.testing.assert_frame_equal(sample_df, ts.data)
