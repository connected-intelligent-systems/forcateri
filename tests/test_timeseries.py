from os import name
import pytest

import numpy as np
import pandas as pd

from forcateri import TimeSeries
from forcateri.data.timeseriesexceptions import (
    InvalidDataFrameFormat,
    InvalidRepresentationFormat,
)


@pytest.fixture
def delta_range():
    return pd.timedelta_range(
        start=0,
        periods=3,
        freq="h",
    )


@pytest.fixture
def time_range():
    return pd.date_range(
        start=pd.Timestamp(2000, 1, 1),
        periods=3,
        freq="h",
    )


@pytest.fixture
def row_index(delta_range, time_range):
    return pd.MultiIndex.from_product(
        [delta_range, time_range],
        names=TimeSeries.ROW_INDEX_NAMES,
    )


@pytest.fixture
def feature_names():
    return ["feat0", "feat1"]


@pytest.fixture
def determ():
    return ["value"]


@pytest.fixture
def quantiles():
    return [0.1, 0.5, 0.9]


@pytest.fixture
def samples():
    return [0, 1, 2]


@pytest.fixture
def determ_col_index(feature_names, determ):
    return pd.MultiIndex.from_product(
        [feature_names, determ],
        names=TimeSeries.COL_INDEX_NAMES,
    )


@pytest.fixture
def quantile_col_index(feature_names, quantiles):
    return pd.MultiIndex.from_product(
        [feature_names, quantiles],
        names=TimeSeries.COL_INDEX_NAMES,
    )


@pytest.fixture
def sample_col_index(feature_names, samples):
    return pd.MultiIndex.from_product(
        [feature_names, samples],
        names=TimeSeries.COL_INDEX_NAMES,
    )


@pytest.fixture
def determ_df(row_index, determ_col_index):
    rows = row_index
    cols = determ_col_index
    data = np.ones(shape=(len(rows), len(cols)))
    df = pd.DataFrame(index=rows, columns=cols, data=data)
    return df


@pytest.fixture
def quantile_df(row_index, quantile_col_index):
    rows = row_index
    cols = quantile_col_index
    data = np.ones(shape=(len(rows), len(cols)))
    df = pd.DataFrame(index=rows, columns=cols, data=data)
    return df


@pytest.fixture
def sample_df(row_index, sample_col_index):
    rows = row_index
    cols = sample_col_index
    data = np.ones(shape=(len(rows), len(cols)))
    df = pd.DataFrame(index=rows, columns=cols, data=data)
    return df


class TestMatchingFormat:

    # positive cases

    def test_matching_determ(self, determ_df):
        assert TimeSeries.is_matching_format(determ_df, TimeSeries.DETERM_REP)

    def test_matching_determ_inference(self, determ_df):
        """
        If no representation argument is passed to is_matching_format,
        TimeSeries.DETERM_REP should be assumed.
        """
        assert TimeSeries.is_matching_format(determ_df)

    def test_matching_quantile(self, quantile_df):
        assert TimeSeries.is_matching_format(quantile_df, TimeSeries.QUANTILE_REP)

    def test_matching_sample(self, sample_df):
        assert TimeSeries.is_matching_format(sample_df, TimeSeries.SAMPLE_REP)

    # negative cases

    def test_malformed_representation_argument(self, determ_df):
        """
        A ValueError should be raised if the representation argument to is_matching_format
        is neither of TimeSeries.DETERM_REP, TimeSeries.QUANTILE_REP, TimeSeries.SAMPLE_REP
        """
        with pytest.raises(ValueError):
            TimeSeries.is_matching_format(determ_df, "unknown_representation")
            TimeSeries.is_matching_format(determ_df, representation=3)

    def test_type_mismatch(self):
        """
        Even a type mismatch should not raise an error but simply return False
        """
        assert not TimeSeries.is_matching_format(
            "not_a_dataframe", TimeSeries.DETERM_REP
        )

    def test_unknown_repr(self, determ_df):
        assert not TimeSeries.is_matching_format(determ_df, "out_of_thin_air")

    def test_malformed_df(self):
        """
        Data frame that doesn't fit the structural criteria
        """
        assert not TimeSeries.is_matching_format(pd.DataFrame(), TimeSeries.DETERM_REP)

    def test_representation_mismatch(self, quantile_df):
        """
        The input data frame has multiple values on the representation axis
        and can therefore not be deterministic.
        """
        assert not TimeSeries.is_matching_format(quantile_df, TimeSeries.DETERM_REP)

    def test_misnamed_rows(self, determ_df):
        determ_df.index.names = ["foo", "bar"]
        assert not TimeSeries.is_matching_format(determ_df, TimeSeries.DETERM_REP)

    def test_misnamed_cols(self, determ_df):
        determ_df.columns.names = ["foo", "bar"]
        assert not TimeSeries.is_matching_format(determ_df, TimeSeries.DETERM_REP)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_swapped_index(self, determ_df, axis):
        determ_df = determ_df.swaplevel(axis=axis)
        assert not TimeSeries.is_matching_format(determ_df, TimeSeries.DETERM_REP)

    def test_mistyped_time_row_index(self, determ_df, delta_range):
        new_index = pd.MultiIndex.from_product(
            [delta_range, ["not", "datetime", "index"]],
            names=TimeSeries.ROW_INDEX_NAMES,
        )
        determ_df.index = new_index
        assert not TimeSeries.is_matching_format(determ_df, TimeSeries.DETERM_REP)

    def test_mistyped_offset_row_index(self, determ_df, time_range):
        """
        Outer row index level must be of type pd.Timedelta
        """
        new_index = pd.MultiIndex.from_product(
            [["not", "timedelta", "index"], time_range],
            names=TimeSeries.ROW_INDEX_NAMES,
        )
        determ_df.index = new_index
        assert not TimeSeries.is_matching_format(determ_df, TimeSeries.DETERM_REP)

    def test_misnamed_determ_col_index(self, determ_df, feature_names):
        """
        All columns must be named "value" at the inner level
        """
        new_index = pd.MultiIndex.from_product(
            [feature_names, ["not_value"]],
            names=TimeSeries.COL_INDEX_NAMES,
        )
        determ_df.columns = new_index
        assert not TimeSeries.is_matching_format(determ_df, TimeSeries.DETERM_REP)

    def test_mistyped_quantile_col_index(self, quantile_df, feature_names):
        """
        Quantile levels must be of type float

        """
        new_index = pd.MultiIndex.from_product(
            [feature_names, [1, 5, 9]],
            names=TimeSeries.COL_INDEX_NAMES,
        )
        quantile_df.columns = new_index
        assert not TimeSeries.is_matching_format(quantile_df, TimeSeries.QUANTILE_REP)

    def test_mistyped_determ_col_index(self, sample_df, feature_names):
        """
        Sample levels must be of type int
        """
        new_index = pd.MultiIndex.from_product(
            [feature_names, [0.1, 0.2, 0.3]],
            names=TimeSeries.COL_INDEX_NAMES,
        )
        sample_df.columns = new_index
        assert not TimeSeries.is_matching_format(sample_df, TimeSeries.DETERM_REP)

    # side effects

    def test_changes_to_df(self, quantile_df):
        """
        Making sure that is_matching_format doesn't change the data frame
        """
        copy = quantile_df.copy()
        assert TimeSeries.is_matching_format(copy, TimeSeries.QUANTILE_REP)
        pd.testing.assert_frame_equal(quantile_df, copy)


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
