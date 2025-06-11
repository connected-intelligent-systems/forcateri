import pandas as pd

from forcateri import TimeSeries

from .common_fixtures import *


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
        assert not TimeSeries.is_matching_format(sample_df, TimeSeries.SAMPLE_REP)

    def test_too_many_determ_values(self, sample_df, feature_names):
        """
        Multiple values per feature should not be allowed in a deterministic TimeSeries
        (This is simuated by converting the sample columns of a well-formed sampled dataframe
        into multiple value columns per feature, resulting in a malformed determ dataframe)
        """
        new_index = pd.MultiIndex.from_product(
            [feature_names, ["value"] * 3],
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
