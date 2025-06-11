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
