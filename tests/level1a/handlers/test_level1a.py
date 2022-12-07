import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd  # type: ignore
import pytest  # type: ignore
from level1a.handlers.level1a import (
    covers,
    get_attitude_records,
    get_ccd_records,
    get_or_raise,
    get_orbit_records,
    get_search_bounds,
    interp_array,
    interpolate,
    parse_event_message,
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


def test_parse_event_message():
    msg = {
        "Records": [{
            "body": json.dumps({
                "Records": [{
                    "s3": {
                        "bucket": {"name": "bucket-name"},
                        "object": {"key": "object-key"}
                    }
                }]
            }),
        }],
    }
    assert parse_event_message(msg) == ("bucket-name", "object-key")


@pytest.mark.parametrize("indices,first,last,expect", (
    (  # returns True when covers
        pd.DatetimeIndex([
            pd.Timestamp(2022, 12, 1),
            pd.Timestamp(2022, 12, 4),
        ]),
        pd.Timestamp(2022, 12, 2),
        pd.Timestamp(2022, 12, 3),
        True,
    ), (  # returns True when first or last is on edge
        pd.DatetimeIndex([
            pd.Timestamp(2022, 12, 1),
            pd.Timestamp(2022, 12, 4),
        ]),
        pd.Timestamp(2022, 12, 1),
        pd.Timestamp(2022, 12, 4),
        True,
    ), (  # returns False when first is early
        pd.DatetimeIndex([
            pd.Timestamp(2022, 12, 1),
            pd.Timestamp(2022, 12, 4),
        ]),
        pd.Timestamp(2022, 11, 2),
        pd.Timestamp(2022, 12, 3),
        False,
    ), (  # returns False when last is late
        pd.DatetimeIndex([
            pd.Timestamp(2022, 12, 1),
            pd.Timestamp(2022, 12, 4),
        ]),
        pd.Timestamp(2022, 12, 2),
        pd.Timestamp(2023, 1, 3),
        False,
    ), (  # returns False when indices is empty
        pd.DatetimeIndex([]),
        pd.Timestamp(2022, 11, 2),
        pd.Timestamp(2022, 12, 3),
        False,
    ),
))
def test_covers(indices, first, last, expect):
    assert covers(indices, first, last) == expect


def test_get_ccd_records(ccd_path):
    out = get_ccd_records(ccd_path)
    expect_inds = pd.DatetimeIndex(
        [
            '2022-11-22 08:32:54.521820068+00:00',
            '2022-11-22 08:33:11.365371704+00:00',
            '2022-11-22 08:33:38.175216675+00:00',
            '2022-11-22 08:35:24.056808472+00:00',
            '2022-11-22 08:35:24.056808472+00:00',
        ],
        dtype='datetime64[ns, UTC]',
        name='EXPDate',
    )

    pd.testing.assert_index_equal(
        out.index,
        expect_inds,
    )


@pytest.mark.parametrize("min_time,max_time,rows", (
    (
        pd.Timestamp(2022, 11, 1),
        pd.Timestamp(2022, 12, 1),
        57230
    ),
    (
        pd.Timestamp(2022, 11, 22, 12),
        pd.Timestamp(2022, 12, 1),
        14105
    ),
    (
        pd.Timestamp(2022, 11, 22, 12),
        pd.Timestamp(2022, 11, 22, 13),
        3598
    ),
))
def test_get_orbit_records(platform_dir, min_time, max_time, rows):
    out = get_orbit_records(platform_dir, min_time, max_time)
    assert list(out.columns) == ["afsTangentPoint", "acsGnssStateJ2000"]
    assert len(out) == rows


@pytest.mark.parametrize("min_time,max_time,rows", (
    (
        pd.Timestamp(2022, 11, 1),
        pd.Timestamp(2022, 12, 1),
        57362
    ),
    (
        pd.Timestamp(2022, 11, 22, 12),
        pd.Timestamp(2022, 12, 1),
        14149
    ),
    (
        pd.Timestamp(2022, 11, 22, 12),
        pd.Timestamp(2022, 11, 22, 13),
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
        pd.Timestamp('2022-11-22 08:32:54.521820068+00:00'),
        pd.Timestamp('2022-11-22 08:35:24.056808472+00:00'),
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
