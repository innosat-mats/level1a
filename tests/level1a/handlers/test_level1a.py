import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd  # type: ignore
import pyarrow.parquet as pq  # type: ignore
import pytest  # type: ignore
from level1a.handlers.level1a import (
    HTR_COLUMNS,
    covers,
    get_attitude_records,
    get_ccd_records,
    get_htr_records,
    get_or_raise,
    get_orbit_records,
    get_search_bounds,
    interp_array,
    interpolate,
    lambda_handler,
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
        pd.Timestamp("2022-11-22T08:00:00+00:00"),
        pd.Timestamp("2022-11-22T10:00:00+00:00"),
        582
    ),
    (
        pd.Timestamp("2022-11-22T08:00:00+00:00"),
        pd.Timestamp("2022-11-22T09:00:00+00:00"),
        338
    ),
    (
        pd.Timestamp("2022-11-22T09:00:00+00:00"),
        pd.Timestamp("2022-11-22T10:00:00+00:00"),
        244
    ),
))
def test_get_htr_records(htr_path, min_time, max_time, rows):
    out = get_htr_records(htr_path, min_time, max_time)
    assert set([*out.columns, out.index.name]) == set(HTR_COLUMNS)
    assert len(out) == rows


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
def test_get_orbit_records(orbit_path, min_time, max_time, rows):
    out = get_orbit_records(orbit_path, min_time, max_time)
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
def test_get_attitude_records(attitude_path, min_time, max_time, rows):
    out = get_attitude_records(attitude_path, min_time, max_time)
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
        '2022-10-01T06:00:00',
        '2022-11-01T06:00:00',
        '2022-11-02T00:00:00',
        '2022-11-02T18:00:00',
        '2022-12-02T18:00:00',
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
            [np.nan, 1.5, 3., 3.75, np.nan],
            index=datetimes,
        ),
    )


def test_interpolate_with_max_diff_returns_nan():
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
        interpolate(dataframe, datetimes, pd.Timedelta(hours=12)),
        pd.DataFrame(
            [1.5, 3., np.nan],
            index=datetimes,
        ),
    )


@patch("level1a.handlers.level1a.pa.fs.S3FileSystem", return_value=None)
@patch.dict(os.environ, {
    "OUTPUT_BUCKET": TemporaryDirectory().name,
    "PLATFORM_BUCKET": str(Path(__file__).parent / "files" / "platform"),
    "HTR_BUCKET": str(Path(__file__).parent / "files" / "rac"),
})
def test_lambda_handler(patched_s3):
    out_dir = os.environ["OUTPUT_BUCKET"]
    (Path(out_dir) / "2022" / "11" / "22").mkdir(parents=True)

    out_file = "2022/11/22/MATS_OPS_Level0_VC1_APID100_20221122-080636_20221122-094142.parquet"  # noqa: E501

    event = {
        "Records": [{
            "body": json.dumps({
                "Records": [{
                    "s3": {
                        "bucket": {
                            "name": str(Path(__file__).parent / "files" / "rac")
                        },
                        "object": {"key": f"CCD/{out_file}"}
                    }
                }]
            }),
        }],
    }

    lambda_handler(event, "")

    df = pq.read_table(f"{out_dir}/{out_file}").to_pandas()
    assert df.index.name == "EXPDate"
    assert set(df.columns) == {
        "OriginFile", "ProcessingTime", "RamsesTime", "QualityIndicator",
        "LossFlag", "VCFrameCounter", "SPSequenceCount", "TMHeaderTime",
        "TMHeaderNanoseconds", "SID", "RID", "CCDSEL", "EXPNanoseconds",
        "WDWMode", "WDWInputDataWindow", "WDWOV", "JPEGQ", "FRAME", "NROW",
        "NRBIN", "NRSKIP", "NCOL", "NCBINFPGAColumns", "NCBINCCDColumns",
        "NCSKIP", "NFLUSH", "TEXPMS", "GAINMode", "GAINTiming",
        "GAINTruncation", "TEMP", "FBINOV", "LBLNK", "TBLNK", "ZERO", "TIMING1",
        "TIMING2", "VERSION", "TIMING3", "NBC", "BadColumns", "ImageName",
        "ImageData", "Warnings", "Errors", "afsAttitudeState",
        "afsTangentPoint", "acsGnssStateJ2000", "HTR1A", "HTR1B", "HTR1OD",
        "HTR2A", "HTR2B", "HTR2OD", "HTR7A", "HTR7B", "HTR7OD", "HTR8A",
        "HTR8B", "HTR8OD",
    }
    assert len(df) == 5
