import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pandas as pd  # type: ignore
import pyarrow.parquet as pq  # type: ignore
import pytest  # type: ignore
from level1a.handlers.level1a import (
    HTR_COLUMNS,
    covers,
    disambiguate_matches,
    dropna_arrays,
    find_match,
    get_level0_records,
    get_htr_records,
    get_mats_schedule_records,
    get_or_raise,
    get_reconstructed_records,
    get_search_bounds,
    interp_array,
    interpolate,
    lambda_handler,
    match_with_schedule,
    parse_event_message,
    MissingSchedule,
    OverlappingSchedulesError,
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


def test_get_level0_records(ccd_path):
    out, meta = get_level0_records(ccd_path, index="EXPDate")
    expect_inds = pd.DatetimeIndex(
        [
            '2022-12-21 23:59:59.063491821+00:00',
            '2022-12-21 23:59:59.063491821+00:00',
            '2022-12-21 23:59:59.063491821+00:00',
            '2022-12-21 23:59:59.370483398+00:00',
        ],
        dtype='datetime64[ns, UTC]',
        name='EXPDate',
    )

    pd.testing.assert_index_equal(
        out.index,
        expect_inds,
    )
    assert meta == {
        b'CODE': b'v1.2.0 (20b4dcc) @ 2022-12-07T09:53:06Z',
        b'RAMSES': b'SPU045-S2:6F',
        b'INNOSAT': b'IS-OSE-ICD-0005:1',
        b'AEZ': b'AEZICD002:I',
    }


def test_get_mats_schedule_records(schedule_path: Path):
    schedule_records = get_mats_schedule_records(
        schedule_path,
        min_time=pd.Timestamp("2023-03-08T00:00:00"),
        max_time=pd.Timestamp("2023-03-11T00:00:00"),
    )

    assert set(schedule_records) == {
        "schedule_description_long", "schedule_description_short",
        "schedule_end_date", "schedule_id", "schedule_name",
        "schedule_pointing_altitudes", "schedule_standard_altitude",
        "schedule_start_date", "schedule_version", "schedule_xml_file",
        "schedule_yaw_correction", "schedule_created_time",
    }
    assert schedule_records.shape == (3, 12)


@pytest.mark.parametrize("min_time,max_time,rows", (
    (
        pd.Timestamp("2022-12-21T13:00:00+00:00"),
        pd.Timestamp("2022-12-22T00:00:00+00:00"),
        3797
    ),
    (
        pd.Timestamp("2022-12-21T13:00:00+00:00"),
        pd.Timestamp("2022-12-21T14:00:00+00:00"),
        224
    ),
    (
        pd.Timestamp("2022-12-21T23:00:00+00:00"),
        pd.Timestamp("2022-12-22T00:00:00+00:00"),
        360
    ),
))
def test_get_htr_records(htr_path, min_time, max_time, rows):
    out = get_htr_records(htr_path, min_time, max_time)
    assert set([*out.columns, out.index.name]) == set(HTR_COLUMNS)
    assert len(out) == rows


@pytest.mark.parametrize("min_time,max_time,rows", (
    (
        pd.Timestamp("2022-12-21T00:00:00Z"),
        pd.Timestamp("2022-12-22T00:00:00Z"),
        11436
    ),
    (
        pd.Timestamp("2022-12-21T20:00:00Z"),
        pd.Timestamp("2022-12-21T21:00:00Z"),
        639
    ),
    (
        pd.Timestamp("2022-12-21T23:00:00Z"),
        pd.Timestamp("2022-12-22T00:00:00Z"),
        3599
    ),
))
def test_get_reconstructed_records(
    reconstructed_path,
    min_time,
    max_time,
    rows,
):
    out = get_reconstructed_records(reconstructed_path, min_time, max_time)
    assert list(out.columns) == [
        "afsAttitudeState",
        "afsGnssStateJ2000",
        "afsTPLongLatGeod",
        "afsTangentH_wgs84",
        "afsTangentPointECI",
    ]
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


@pytest.mark.parametrize("csv_data, expected", [
    [  # Select higher version
        {
            "schedule_start_date": [
                "2022-12-19 18:00:00",
                "2022-12-19 18:00:00",
            ],
            "schedule_end_date": [
                "2022-12-19 23:59:59",
                "2022-12-19 23:59:59",
            ],
            "schedule_id": [3105, 3105],
            "schedule_name": ["MODE1y", "MODE1y"],
            "schedule_version": [2, 1],
            "schedule_standard_altitude": [92500, 92500],
            "schedule_yaw_correction": [True, True],
            "schedule_pointing_altitudes": [[], []],
            "schedule_xml_file": [
                "STP-MTS-3105_22121922121902TMODE1y.xml",
                "STP-MTS-3106_22121922121901TMODE1y.xml",
            ],
            "schedule_description_short": [
                "Operational mode",
                "Operational mode",
            ],
            "schedule_description_long": [
                "Mode 1. JPEGQ = 90  (should have been 80)",
                "Mode 1. JPEGQ = 90  (should have been 80)",
            ],
        },
        "STP-MTS-3105_22121922121902TMODE1y.xml",
    ],
    [  # Select latest generation date
        {
            "schedule_start_date": ["2023-04-13", "2023-04-13"],
            "schedule_end_date": [
                "2023-04-13 23:59:56",
                "2023-04-13 23:59:56",
            ],
            "schedule_id": [1207, 1207],
            "schedule_name": ["CROPD", "CROPD"],
            "schedule_version": [3, 3],
            "schedule_standard_altitude": [87500, 87500],
            "schedule_yaw_correction": [True, True],
            "schedule_pointing_altitudes": [[], []],
            "schedule_xml_file": [
                "STP-MTS-1207_23041323041103TCROPD.xml",
                "STP-MTS-1207_23041323041003TCROPD.xml",
            ],
            "schedule_description_short": ["", ""],
            "schedule_description_long": ["", ""],
        },
        "STP-MTS-1207_23041323041103TCROPD.xml",
    ]
])
def test_disambiguate_matches(csv_data: dict, expected: str):
    dataframe = pd.DataFrame.from_dict(csv_data)
    matches = disambiguate_matches(dataframe)
    pd.testing.assert_frame_equal(
        matches,
        dataframe[dataframe["schedule_xml_file"] == expected],
    )


@pytest.mark.parametrize("csv_data, expected", [
    [
        {
            "schedule_pointing_altitudes": [42, 43],
            "schedule_xml_file": [
                "STP-MTS-1207_23041323041103TCROPD.xml",
                "STP-MTS-1207_23041323041103TCROPD.xml",
            ],
        },
        "column schedule_pointing_altitudes differs"
    ],
    [
        {
            "schedule_pointing_altitudes": [42, 42],
            "schedule_xml_file": [
                "STP-MTS-1207_23041323041103TCROPD.xml",
                "STP-MTS-1207_23041223041103TCROPD.xml",
            ],
        },
        "execution dates differ",
    ],
])
def test_disambiguate_matches_raises(csv_data: Dict[str, Any], expected: str):
    dataframe = pd.DataFrame.from_dict(csv_data)
    with pytest.raises(OverlappingSchedulesError, match=expected):
        _ = disambiguate_matches(dataframe)


def test_find_match(schedule: pd.DataFrame):
    target_date = pd.DatetimeIndex(["1978-03-29T22:42:00"])[0]
    answer = find_match(
        target_date=target_date,
        column="Answer",
        dataframe=schedule,
    )
    assert answer == 42


def test_find_match_with_buffer(schedule: pd.DataFrame):
    target_date = pd.DatetimeIndex(["1978-03-29T23:30:00"])[0]
    answer = find_match(
        target_date=target_date,
        column="Answer",
        dataframe=schedule,
        buffer=pd.Timedelta(hours=1),
    )
    assert answer == 42


def test_find_match_warns_on_missing(schedule: pd.DataFrame):
    target_date = pd.DatetimeIndex(["1978-03-29T23:30:00"])[0]
    with pytest.warns(MissingSchedule, match="Missing schedule"):
        find_match(
            target_date=target_date,
            column="Answer",
            dataframe=schedule,
        )


def test_find_match_warns_on_missing_with_buffer(schedule: pd.DataFrame):
    target_date = pd.DatetimeIndex(["1978-03-29T23:30:00"])[0]
    with pytest.warns(MissingSchedule, match="Missing schedule"):
        find_match(
            target_date=target_date,
            column="Answer",
            dataframe=schedule,
            buffer=pd.Timedelta(minutes=1),
        )


def test_find_match_does_not_raise_on_identical_overlaps(
    schedule: pd.DataFrame,
):
    target_date = pd.DatetimeIndex(["2010-10-10T10:10:10"])[0]
    fit = find_match(
        target_date=target_date,
        column="Fit",
        dataframe=schedule,
    )
    assert fit == -1


def test_find_match_selects_latest_file(schedule: pd.DataFrame):
    target_date = pd.DatetimeIndex(["2010-10-10T10:10:10"])[0]
    schedule["schedule_created_time"] = [0, 0, 0, 0, 0, 0, 0, 1]
    answer = find_match(
        target_date=target_date,
        column="Answer",
        dataframe=schedule,
    )
    assert answer == -2


def test_match_with_schedule(schedule: pd.DataFrame):
    target = pd.DatetimeIndex([
        "1978-03-08T22:42:00",
        "1978-03-15T22:42:00",
        "1978-03-22T22:42:00",
        "1978-03-29T22:42:00",
        "1978-04-05T22:42:00",
        "1978-04-12T22:42:00",
    ])
    matched = match_with_schedule(schedule, target)
    pd.testing.assert_frame_equal(matched, schedule[:6].set_index(target))


def test_dropna_arrays():
    dataframe = pd.DataFrame.from_dict({
        "Squared": [[np.nan, 1764], [1764, 1764]],
        "Answer": [-42, 42],
    })
    pd.testing.assert_frame_equal(
        dropna_arrays(dataframe, columns=["Squared"]),
        dataframe[dataframe.Answer == 42],
    )


@patch("level1a.handlers.level1a.pa.fs.S3FileSystem", return_value=None)
@patch.dict(os.environ, {
    "OUTPUT_BUCKET": TemporaryDirectory().name,
    "PLATFORM_BUCKET": str(Path(__file__).parent / "files" / "platform"),
    "MATS_SCHEDULE_BUCKET": str(Path(__file__).parent / "files" / "schedule"),
    "HTR_BUCKET": str(Path(__file__).parent / "files" / "rac"),
    "L1A_VERSION": "latest.and.greatest",
    "DATA_PREFIX": "CCD",
    "TIME_COLUMN": "EXPDate",
})
@pytest.mark.filterwarnings("ignore:Discarding nonzero nanoseconds")
def test_lambda_handler(patched_s3):
    out_dir = os.environ["OUTPUT_BUCKET"]
    (Path(out_dir) / "2022" / "12" / "21" / "23").mkdir(parents=True)

    out_file = "2022/12/21/23/MATS_OPS_Level0_VC1_APID100_20221221-132606_20221222-133247.parquet"  # noqa: E501

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
    assert set(df.columns) == {
        "EXPDate", "OriginFile", "ProcessingTime", "RamsesTime",
        "QualityIndicator", "LossFlag", "VCFrameCounter", "SPSequenceCount",
        "TMHeaderTime", "TMHeaderNanoseconds", "SID", "RID", "CCDSEL",
        "EXPNanoseconds", "WDWMode", "WDWInputDataWindow", "WDWOV", "JPEGQ",
        "FRAME", "NROW", "NRBIN", "NRSKIP", "NCOL", "NCBINFPGAColumns",
        "NCBINCCDColumns", "NCSKIP", "NFLUSH", "TEXPMS", "GAINMode",
        "GAINTiming", "GAINTruncation", "TEMP", "FBINOV", "LBLNK", "TBLNK",
        "ZERO", "TIMING1", "TIMING2", "VERSION", "TIMING3", "NBC", "BadColumns",
        "ImageName", "ImageData", "Warnings", "Errors", "afsAttitudeState",
        "afsGnssStateJ2000", "afsTPLongLatGeod", "afsTangentH_wgs84",
        "afsTangentPointECI", "HTR1A", "HTR1B", "HTR1OD", "HTR2A", "HTR2B",
        "HTR2OD", "HTR7A", "HTR7B", "HTR7OD", "HTR8A", "HTR8B", "HTR8OD",
        "satlat", "satlon", "satheight", "TPlat", "TPlon", "TPheight", "TPsza",
        "TPssa", "nadir_sza", "nadir_az", "TPlocaltime",
        "schedule_created_time", "schedule_description_long",
        "schedule_description_short", "schedule_end_date", "schedule_id",
        "schedule_name", "schedule_pointing_altitudes",
        "schedule_standard_altitude", "schedule_start_date", "schedule_version",
        "schedule_xml_file", "schedule_yaw_correction", "channel", "id",
        "flipped", "temperature_ADC", "temperature", "temperature_HTR",
        "RAMSES", "AEZ", "L1ADataPath", "L1ADataBucket", "RACCode", "L1ACode",
        "DataLevel", "INNOSAT",
    }
    assert len(df) == 4


@patch("level1a.handlers.level1a.pa.fs.S3FileSystem", return_value=None)
@patch.dict(os.environ, {
    "OUTPUT_BUCKET": TemporaryDirectory().name,
    "PLATFORM_BUCKET": str(Path(__file__).parent / "files" / "platform"),
    "MATS_SCHEDULE_BUCKET": str(Path(__file__).parent / "files" / "schedule"),
    "L1A_VERSION": "latest.and.greatest",
    "DATA_PREFIX": "PM",
    "TIME_COLUMN": "PMTime",
})
def test_lambda_handler_no_htr(patched_s3):
    out_dir = os.environ["OUTPUT_BUCKET"]
    (Path(out_dir) / "2022" / "12" / "21").mkdir(parents=True)

    out_file = "2022/12/21/MATS_OPS_Level0_VC1_APID100_20221221-132606_20221222-133247.parquet"  # noqa: E501

    event = {
        "Records": [{
            "body": json.dumps({
                "Records": [{
                    "s3": {
                        "bucket": {
                            "name": str(Path(__file__).parent / "files" / "rac")
                        },
                        "object": {"key": f"PM/{out_file}"}
                    }
                }]
            }),
        }],
    }

    lambda_handler(event, "")

    df = pq.read_table(f"{out_dir}/{out_file}").to_pandas()
    assert set(df.columns) == {
        "Errors", "LossFlag", "OriginFile", "PM1A", "PM1ACNTR", "PM1B",
        "PM1BCNTR", "PM1S", "PM1SCNTR", "PM2A", "PM2ACNTR", "PM2B", "PM2BCNTR",
        "PM2S", "PM2SCNTR", "PMNanoseconds", "PMTime", "ProcessingTime",
        "QualityIndicator", "RID", "RamsesTime", "SID", "SPSequenceCount",
        "TMHeaderNanoseconds", "TMHeaderTime", "TPheight", "TPlat",
        "TPlocaltime", "TPlon", "TPssa", "TPsza", "VCFrameCounter", "Warnings",
        "afsAttitudeState", "afsGnssStateJ2000", "afsTPLongLatGeod",
        "afsTangentH_wgs84", "afsTangentPointECI", "index", "nadir_sza",
        "nadir_az", "satheight", "satlat", "satlon", "schedule_created_time",
        "schedule_description_long", "schedule_description_short",
        "schedule_end_date", "schedule_id", "schedule_name",
        "schedule_pointing_altitudes", "schedule_standard_altitude",
        "schedule_start_date", "schedule_version", "schedule_xml_file",
        "schedule_yaw_correction", "L1ADataPath", "L1ADataBucket", "L1ACode",
        "DataLevel",
    }
    assert len(df) == 8
