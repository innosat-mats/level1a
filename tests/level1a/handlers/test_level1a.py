import os
from datetime import timezone
from unittest.mock import patch

import numpy as np
import pandas as pd  # type: ignore
import pytest  # type: ignore
from level1a.handlers.level1a import (
    get_attitude_records,
    get_ccd_records,
    get_filename,
    get_last_date,
    get_or_raise,
    get_orbit_records,
    get_partitioned_dates,
    get_search_bounds,
    interp_array,
    interpolate,
)


@patch.dict(os.environ, {"DEFINITELY": "set"})
def test_get_or_raise():
    assert get_or_raise("DEFINITELY") == "set"


def test_get_or_raise_raises():
    with pytest.raises(
        EnvironmentError,
        match="DEFINITELYNOT is a required environment variable"
    ):
        get_or_raise("DEFINITELYNOT")


@pytest.mark.parametrize("start,expect", (
    (
        pd.Timestamp(2022, 11, 1, tzinfo=timezone.utc),
        pd.Timestamp('2022-11-22 08:35:24.056808472+0000', tz='UTC')
    ),
    (  # No valid dates returns None
        pd.Timestamp(2022, 12, 1, tzinfo=timezone.utc),
        None
    ),
))
def test_get_last_date(output_dataset, start, expect):
    assert get_last_date(output_dataset, start) == expect


def test_get_last_date_handles_empty_dataset(empty_dataset):
    assert get_last_date(
        empty_dataset,
        pd.Timestamp(2022, 1, 1, tzinfo=timezone.utc)
    ) is None


@pytest.mark.parametrize("min_time,inds", (
    (
        pd.Timestamp(2022, 11, 1, tzinfo=timezone.utc),
        [
            '2022-11-22 08:32:54.521820068+00:00',
            '2022-11-22 08:33:11.365371704+00:00',
            '2022-11-22 08:33:38.175216675+00:00',
            '2022-11-22 08:35:24.056808472+00:00',
            '2022-11-22 08:35:24.056808472+00:00',
        ],
    ),
    (
        pd.Timestamp(2022, 11, 22, 8, 34, tzinfo=timezone.utc),
        [
            '2022-11-22 08:35:24.056808472+00:00',
            '2022-11-22 08:35:24.056808472+00:00',
        ],
    ),
))
def test_get_ccd_records(rac_dir, min_time, inds):
    out = get_ccd_records(rac_dir, min_time)
    pd.testing.assert_index_equal(
        out.index,
        pd.DatetimeIndex(inds, dtype='datetime64[ns, UTC]', name='EXPDate')
    )


@pytest.mark.parametrize("min_time,max_time,rows", (
    (
        np.datetime64("2022-11-01T00:00:00"),
        np.datetime64("2022-12-01T00:00:00"),
        57230
    ),
    (
        np.datetime64("2022-11-22T12:00:00"),
        np.datetime64("2022-12-01T00:00:00"),
        14105
    ),
    (
        np.datetime64("2022-11-22T12:00:00"),
        np.datetime64("2022-11-22T13:00:00"),
        3598
    ),
))
def test_get_orbit_records(platform_dir, min_time, max_time, rows):
    out = get_orbit_records(platform_dir, min_time, max_time)
    assert list(out.columns) == ["afsTangentPoint", "acsGnssStateJ2000"]
    assert len(out) == rows


@pytest.mark.parametrize("min_time,max_time,rows", (
    (
        np.datetime64("2022-11-01T00:00:00"),
        np.datetime64("2022-12-01T00:00:00"),
        57362
    ),
    (
        np.datetime64("2022-11-22T12:00:00"),
        np.datetime64("2022-12-01T00:00:00"),
        14149
    ),
    (
        np.datetime64("2022-11-22T12:00:00"),
        np.datetime64("2022-11-22T13:00:00"),
        3603
    ),
))
def test_get_attitude_records(platform_dir, min_time, max_time, rows):
    out = get_attitude_records(platform_dir, min_time, max_time)
    assert list(out.columns) == ["afsAttitudeState"]
    assert len(out) == rows


def test_get_search_bounds():
    timeinds = pd.DatetimeIndex([
        '2022-11-22 08:35:24.056808472+00:00',
        '2022-11-22 08:32:54.521820068+00:00',
        '2022-11-22 08:33:11.365371704+00:00',
        '2022-11-22 08:35:24.056808472+00:00',
        '2022-11-22 08:33:38.175216675+00:00',
    ])
    assert get_search_bounds(timeinds) == (
        np.datetime64('2022-11-22 08:32:24.521820068+00:00'),
        np.datetime64('2022-11-22 08:35:54.056808472+00:00'),
    )


@pytest.mark.parametrize("eval_point,indices,values,expect", (
    (
        1,
        np.array([1, 2]),
        np.array([1, 2]),
        1,
    ),
    (
        2,
        np.array([1, 3]),
        np.array([[1, 10, 100], [3, 30, 300]]),
        np.array([2, 20, 200]),
    ),
    (
        3,
        np.array([3, 3]),
        np.array([10, 10]),
        10,
    )
))
def test_interp_array(eval_point, indices, values, expect):
    np.testing.assert_array_equal(
        interp_array(eval_point, indices, values),
        expect,
    )


def test_interpolate():
    datetimes = pd.DatetimeIndex([
        '2022-11-01T06:00:00',
        '2022-11-02T00:00:00',
        '2022-11-02T18:00:00',
    ])
    dataframe = pd.DataFrame(
        [1., 2., 3., 4.],
        index=pd.DatetimeIndex([
            '2022-11-01T00:00:00',
            '2022-11-01T12:00:00',
            '2022-11-02T00:00:00',
            '2022-11-03T00:00:00',
        ])
    )
    pd.testing.assert_frame_equal(
        interpolate(dataframe, datetimes),
        pd.DataFrame(
            [1.5, 3., 3.75],
            index=datetimes,
        )
    )


def test_get_partitioned_dates():
    datetimes = pd.DatetimeIndex([
        '2022-11-01',
        '2022-11-02',
        '2022-11-03',
    ])
    pd.testing.assert_frame_equal(
        get_partitioned_dates(datetimes),
        pd.DataFrame({
            'year': [2022, 2022, 2022],
            'month': [11, 11, 11],
            'day': [1, 2, 3],
        }, index=datetimes)
    )


def test_get_filename():
    assert get_filename(pd.DatetimeIndex([
        '2022-11-01',
        '2022-11-02',
        '2022-11-03',
    ])) == "payload-level1a_20221101-000000_20221103-000000_{i}.parquet"
