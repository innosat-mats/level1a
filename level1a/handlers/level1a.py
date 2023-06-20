import json
import os
import warnings
from functools import wraps
from http import HTTPStatus
from time import sleep
from traceback import format_tb
from typing import Any, Callable, Dict, Optional, Set, Tuple

import numpy as np
import pyarrow as pa  # type: ignore
import pyarrow.dataset as ds  # type: ignore
import pyarrow.parquet as pq  # type: ignore
from pandas import (  # type: ignore
    DataFrame,
    DatetimeIndex,
    Series,
    Timedelta,
    Timestamp,
    concat,
    to_datetime,
)
from skyfield.api import load  # type: ignore

try:
    from mats_utils.coordinates import (  # type: ignore
        eci_to_latlon,
        local_time,
        solar_angles,
    )
    from mats_utils.ccd_item import add_ccd_item_attributes  # type: ignore
except ImportError:
    from .mats_utils.coordinates import (
        eci_to_latlon,
        local_time,
        solar_angles,
    )
    from .mats_utils.ccd_item import add_ccd_item_attributes


Event = Dict[str, Any]
Context = Any

RETRIES = 5
OFFSET_FACTOR = 2
RECONSTRUCTED_FREQUENCY = 1.  # Hz
HTR_FREQUENCY = 0.1  # Hz

HTR_COLUMNS = [
    "TMHeaderTime", "HTR1A", "HTR1B", "HTR1OD", "HTR2A", "HTR2B", "HTR2OD",
    "HTR7A", "HTR7B", "HTR7OD", "HTR8A", "HTR8B", "HTR8OD",
]

PAYLOAD_PARTITIONS = pa.schema([
    ("year", pa.int16()),
    ("month", pa.int8()),
    ("day", pa.int8()),
])

PLATFORM_PARTITIONS = pa.schema([
    ("year", pa.int16()),
    ("month", pa.int8()),
    ("day", pa.int8()),
])

SCHEDULE_PARTITIONS = pa.schema([
    ("created_time", pa.int32()),
])
SCHEDULE_BUFFER = Timedelta(seconds=60)
DUMMY_SCHEDULE: Dict[str, Any] = {
    "schedule_created_time": 0,
    "schedule_start_date": to_datetime(0),
    "schedule_end_date": to_datetime(0),
    "schedule_id": 0,
    "schedule_name": "NONE",
    "schedule_version": 0,
    "schedule_standard_altitude": 0,
    "schedule_yaw_correction": False,
    "schedule_pointing_altitudes": "[]",
    "schedule_xml_file": "",
    "schedule_description_short": "",
    "schedule_description_long": "[]",
    "Answer": -42,
}


class DoesNotCover(Exception):
    pass


class InvalidMessage(Exception):
    pass


class Level1AException(Exception):
    pass


class RetriesExceeded(Exception):
    pass


class OverlappingSchedulesError(Exception):
    pass


class OverlappingSchedulesWarning(Warning):
    pass


class MissingSchedule(Warning):
    pass


def s3_backoff(caller: Callable):
    @wraps(caller)
    def wrapper(*args, **kwargs):
        msg = ""
        for r in range(RETRIES + 1):
            try:
                sleep(2 ** r - 1)
                return caller(*args, **kwargs)
            except Exception as err:
                msg = str(err)
        raise RetriesExceeded(msg)
    return wrapper


def get_or_raise(variable_name: str) -> str:
    if (var := os.environ.get(variable_name)) is None:
        raise EnvironmentError(
            f"{variable_name} is a required environment variable"
        )
    return var


def parse_event_message(event: Event) -> Tuple[str, str]:
    try:
        message: Dict[str, Any] = json.loads(event["Records"][0]["body"])
        bucket = message["Records"][0]["s3"]["bucket"]["name"]
        key = message["Records"][0]["s3"]["object"]["key"]
    except (KeyError, TypeError):
        raise InvalidMessage
    return bucket, key


def covers(
    indices: DatetimeIndex,
    first: Timestamp,
    last: Timestamp,
) -> bool:
    return (
        len(indices) != 0
        and first >= indices.min()
        and last <= indices.max()
    )


@s3_backoff
def get_level0_records(
    path_or_bucket: str,
    index: str,
    filesystem: pa.fs.FileSystem = None,
) -> Tuple[DataFrame, pq.FileMetaData]:
    table = pq.read_table(
        path_or_bucket,
        filesystem=filesystem,
    )
    return (
        table.to_pandas().set_index(index).sort_index(),
        table.schema.metadata,
    )


@s3_backoff
def get_mats_schedule_records(
    path_or_bucket: str,
    min_time: Timestamp,
    max_time: Timestamp,
    filesystem: pa.fs.FileSystem = None,
) -> DataFrame:
    dataset = ds.dataset(
        path_or_bucket,
        filesystem=filesystem,
        partitioning=ds.partitioning(SCHEDULE_PARTITIONS, flavor="filename"),
    ).to_table(
        filter=(
            (ds.field("start_date") <= max_time.asm8)
            & (ds.field("end_date") >= min_time.asm8)
        )
    ).to_pandas()
    new_columns = {c: f"schedule_{c}" for c in dataset.columns}
    dataset.rename(columns=new_columns, inplace=True)
    return dataset


@s3_backoff
def get_htr_records(
    path_or_bucket: str,
    min_time: Timestamp,
    max_time: Timestamp,
    filesystem: pa.fs.FileSystem = None,
) -> DataFrame:
    dataset = ds.dataset(
        path_or_bucket,
        filesystem=filesystem,
        partitioning=ds.partitioning(PAYLOAD_PARTITIONS),
    ).to_table(
        filter=(
            (ds.field('year') >= min_time.year)
            & (ds.field('year') <= max_time.year)
            & (
                (
                    (ds.field('month') >= min_time.month)
                    & (ds.field('month') <= max_time.month)
                ) | (
                    (ds.field('year') == min_time.year)
                    & (ds.field('month') >= min_time.month)
                ) | (
                    (ds.field('year') == max_time.year)
                    & (ds.field('month') <= max_time.month)
                )
            )
            & (
                (
                    (ds.field('day') >= min_time.day)
                    & (ds.field('day') <= max_time.day)
                ) | (
                    (ds.field('month') == min_time.month)
                    & (ds.field('day') >= min_time.day)
                ) | (
                    (ds.field('month') == max_time.month)
                    & (ds.field('day') <= max_time.day)
                )
            )
        ),
        columns=HTR_COLUMNS,
    ).to_pandas().drop_duplicates("TMHeaderTime")
    dataset = dataset[
        (dataset["TMHeaderTime"] >= min_time)
        & (dataset["TMHeaderTime"] <= max_time)
    ]
    dataset = dataset.set_index("TMHeaderTime").sort_index()
    return dataset


@s3_backoff
def get_reconstructed_records(
    path_or_bucket: str,
    min_time: Timestamp,
    max_time: Timestamp,
    filesystem: pa.fs.FileSystem = None,
) -> DataFrame:
    dataset = ds.dataset(
        path_or_bucket,
        filesystem=filesystem,
        schema=pa.schema([
            ("time", pa.timestamp('ns')),
            ("afsAttitudeState", pa.list_(pa.float64())),
            ("afsGnssStateJ2000", pa.list_(pa.float64())),
            ("afsTPLongLatGeod", pa.list_(pa.float64())),
            ("afsTangentH_wgs84", pa.list_(pa.float64())),
            ("afsTangentPointECI", pa.list_(pa.float64())),
            ("year", pa.int16()),
            ("month", pa.int8()),
            ("day", pa.int8()),
        ]),
        partitioning=ds.partitioning(PLATFORM_PARTITIONS),
    ).to_table(filter=(
        (ds.field('year') >= min_time.year)
        & (ds.field('year') <= max_time.year)
        & (
            (
                (ds.field('month') >= min_time.month)
                & (ds.field('month') <= max_time.month)
            ) | (
                (ds.field('year') == min_time.year)
                & (ds.field('month') >= min_time.month)
            ) | (
                (ds.field('year') == max_time.year)
                & (ds.field('month') <= max_time.month)
            )
        )
        & (
            (
                (ds.field('day') >= min_time.day)
                & (ds.field('day') <= max_time.day)
            ) | (
                (ds.field('month') == min_time.month)
                & (ds.field('day') >= min_time.day)
            ) | (
                (ds.field('month') == max_time.month)
                & (ds.field('day') <= max_time.day)
            )
        )
    )).to_pandas().drop_duplicates("time")
    dataset = dataset[
        (dataset["time"] >= min_time.asm8)
        & (dataset["time"] <= max_time.asm8)
    ]
    dataset = dataset.set_index("time").sort_index()
    dataset.index = dataset.index.tz_localize('utc')
    dataset.drop(columns=["year", "month", "day"], inplace=True)
    return dataset


def get_search_bounds(
    timeinds: DatetimeIndex
) -> Tuple[Timestamp, Timestamp]:
    return timeinds.min(), timeinds.max()


def get_offset(frequency: float) -> Timedelta:
    return OFFSET_FACTOR * Timedelta(seconds=1 / frequency)


def interp_array(
    eval_ind: float,
    indices: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    diffs = np.abs(eval_ind - indices)
    if np.min(diffs) == 0:
        return values[np.argmin(diffs)]
    dt = np.diff(indices)
    weights = diffs[::-1] / dt
    return np.average(values, weights=weights, axis=0)


def interp_to(
    target_date: Timestamp,
    column: Series,
    max_diff: Optional[Timedelta] = None,
) -> np.ndarray:
    before = column.index.get_indexer([target_date], method='ffill')[0]
    after = column.index.get_indexer([target_date], method='bfill')[0]

    if (before < 0 or after < 0) or (before > after):
        return column.iloc[0] * np.nan

    timestamps = column.index[[before, after]]

    if (
        max_diff is not None
        and np.diff(timestamps)[0] > max_diff
    ):
        return column.iloc[0] * np.nan

    return interp_array(
        target_date.value,
        timestamps.astype(int).values,
        column.iloc[[before, after]].values,
    )


def interpolate(
    dataframe: DataFrame,
    target: DatetimeIndex,
    max_diff: Optional[Timedelta] = None,
) -> DataFrame:
    return DataFrame({
        column: [interp_to(ind, dataframe[column], max_diff) for ind in target]
        for column in dataframe
    }, index=target)


def disambiguate_matches(matches: DataFrame) -> Any:
    generation_dates: Set[str] = set()
    execution_dates: Set[str] = set()
    versions: Set[str] = set()
    for data in matches["schedule_xml_file"]:
        temp = data.split("_")[1]
        execution_dates = execution_dates.union({temp[0: 6]})
        generation_dates = generation_dates.union({temp[6: 12]})
        versions = versions.union({temp[12: 14]})

    if len(execution_dates) != 1:
        msg = f"execution dates differ for interval: {execution_dates}"
        raise OverlappingSchedulesError(msg)
    execution_date = list(execution_dates)[0]

    if len(generation_dates) != 1:
        desired = "_" + execution_date + sorted(generation_dates)[-1]
    elif len(versions) != 1:
        generation_date = list(generation_dates)[0]
        desired = "_" + execution_date + generation_date + sorted(versions)[-1]
    else:
        desired = ""

    matches = matches[matches["schedule_xml_file"].str.contains(desired)]
    if len(matches) != 1:
        for column in matches.columns:
            if not (matches[column].apply(
                lambda x: x == matches[column][0])
            ).all():
                msg = f"column {column} differs for interval"
                raise OverlappingSchedulesError(msg)
        msg = "unknown problem for interval"
        raise OverlappingSchedulesError(msg)

    return matches.reset_index()


def find_match(
    target_date: Timestamp,
    column: str,
    dataframe: DataFrame,
    buffer: Optional[Timedelta] = None,
) -> Any:
    target_date.floor
    matches = dataframe[
        (dataframe["schedule_start_date"] <= target_date.asm8)
        & (dataframe["schedule_end_date"] >= target_date.floor('s').asm8)
    ].reset_index()

    if len(matches) > 1:
        matches = matches[
            matches["schedule_created_time"]
            == matches["schedule_created_time"].max()
        ].reset_index()
        if not (matches[column][0] == matches[column]).all():
            msg = f"Overlapping schedules for target date {target_date}"
            try:
                matches = disambiguate_matches(matches)
                warnings.warn(msg, OverlappingSchedulesWarning)
            except OverlappingSchedulesError as err:
                raise OverlappingSchedulesError(f"{msg}: {err}")
    elif len(matches) == 0:
        if buffer is not None:
            return find_match(
                target_date - buffer,
                column,
                dataframe,
                buffer=None,
            )
        msg = f"Missing schedule for target date {target_date}"
        warnings.warn(msg, MissingSchedule)
        return DUMMY_SCHEDULE[column]

    return matches[column][0]


def match_with_schedule(
    dataframe: DataFrame,
    target: DatetimeIndex,
    buffer: Optional[Timedelta] = None,
) -> DataFrame:
    return DataFrame({
        column: [
            find_match(ind, column, dataframe, buffer)
            for ind in target
        ]
        for column in dataframe
    }, index=target)


def add_satellite_position_data(
    dataframe: DataFrame,
    index: str,
) -> DataFrame:
    dataframe.reset_index(inplace=True)
    timescale = load.timescale()

    dataframe[["satlat", "satlon", "satheight"]] = dataframe.apply(
        lambda s: eci_to_latlon(
            timescale.from_datetime(s[index].to_pydatetime()),
            s.afsGnssStateJ2000[:3],
        ),
        axis=1,
        result_type="expand",
    )

    dataframe[["TPlat", "TPlon", "TPheight"]] = dataframe.apply(
        lambda s: eci_to_latlon(
            timescale.from_datetime(s[index].to_pydatetime()),
            s.afsTangentPointECI,
        ),
        axis=1,
        result_type="expand",
    )

    dataframe[["nadir_sza", "TPsza", "TPssa", "nadir_az"]] = dataframe.apply(
        lambda s: solar_angles(
            timescale.from_datetime(s[index].to_pydatetime()),
            s.satlat, s.satlon, s.satheight,
            s.TPlat, s.TPlon, s.TPheight,
        ),
        axis=1,
        result_type="expand",
    )

    dataframe["TPlocaltime"] = dataframe.apply(
        lambda s: local_time(
            timescale.from_datetime(s[index].to_pydatetime()),
            s.TPlon,
        ),
        axis=1,
        result_type="expand",
    )

    return dataframe.set_index(index).sort_index()


def lambda_handler(event: Event, context: Context):
    try:
        output_bucket = get_or_raise("OUTPUT_BUCKET")
        platform_bucket = get_or_raise("PLATFORM_BUCKET")
        mats_schedule_bucket = get_or_raise("MATS_SCHEDULE_BUCKET")
        code_version = get_or_raise("L1A_VERSION")
        data_prefix = get_or_raise("DATA_PREFIX")
        time_column = get_or_raise("TIME_COLUMN")
        region = os.environ.get('AWS_REGION', "eu-north-1")
        htr_bucket = os.environ.get("HTR_BUCKET", None)
        s3 = pa.fs.S3FileSystem(region=region)

        try:
            bucket, object_path = parse_event_message(event)
            output_path = object_path.strip(f"/{data_prefix}")
        except InvalidMessage:
            return {
                'statusCode': HTTPStatus.NO_CONTENT,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'message': 'Failed to parse event, nothing to do.'
                })
            }

        if not object_path.endswith(".parquet"):
            return {
                'statusCode': HTTPStatus.NO_CONTENT,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'message': f'{object_path} is not a parquet file, nothing to do.'  # noqa: E501
                })
            }

        rac_df, metadata = get_level0_records(
            f"{bucket}/{object_path}",
            index=time_column,
            filesystem=s3,
        )
        metadata.update({
            "L1ACode": code_version,
            "DataLevel": "L1A",
            "L1ADataBucket": output_bucket,
            "L1ADataPath": output_path,
        })
        if "CODE" in metadata.keys():
            metadata["RACCode"] = metadata.pop("CODE")
        elif b"CODE" in metadata.keys():
            metadata["RACCode"] = metadata.pop(b"CODE")
        if "pandas" in metadata.keys():
            del metadata["pandas"]
        elif b"pandas" in metadata.keys():
            del metadata[b"pandas"]

        min_time, max_time = get_search_bounds(rac_df.index)
    except Exception as err:
        tb = '|'.join(format_tb(err.__traceback__)).replace('\n', ';')
        msg = f"Failed to initialize handler: {err} ({type(err)}; {tb})"
        raise Level1AException(msg)

    try:
        reconstructed_df = get_reconstructed_records(
            f"{platform_bucket}/ReconstructedData",
            min_time - get_offset(RECONSTRUCTED_FREQUENCY),
            max_time + get_offset(RECONSTRUCTED_FREQUENCY),
            filesystem=s3,
        )

        if not covers(reconstructed_df.index, min_time, max_time):
            raise DoesNotCover("Reconstructed data is missing timestamps")

        reconstructed_df = interpolate(
            reconstructed_df,
            rac_df.index,
            max_diff=get_offset(RECONSTRUCTED_FREQUENCY),
        )
        reconstructed_df = add_satellite_position_data(
            reconstructed_df,
            index=time_column,
        )
    except Exception as err:
        tb = '|'.join(format_tb(err.__traceback__)).replace('\n', ';')
        msg = f"Failed to get reconstructed data for {output_path} with start time {min_time} and end time {max_time}: {err} ({type(err)}; {tb})"  # noqa: E501
        raise Level1AException(msg)

    try:
        schedule_df = get_mats_schedule_records(
            mats_schedule_bucket,
            min_time,
            max_time,
            filesystem=s3,
        )
        matched_schedule = match_with_schedule(
            schedule_df,
            rac_df.index,
            buffer=SCHEDULE_BUFFER,
        )
    except Exception as err:
        tb = '|'.join(format_tb(err.__traceback__)).replace('\n', ';')
        msg = f"Failed to get schedule data for {output_path} with start time {min_time} and end time {max_time}: {err} ({type(err)}; {tb})"  # noqa: E501
        raise Level1AException(msg)

    if htr_bucket is not None:
        try:
            htr_df = get_htr_records(
                f"{htr_bucket}/HTR",
                min_time - get_offset(HTR_FREQUENCY),
                max_time + get_offset(HTR_FREQUENCY),
                filesystem=s3,
            )
            htr_subset = interpolate(
                htr_df,
                rac_df.index,
                max_diff=get_offset(HTR_FREQUENCY),
            )
        except Exception as err:
            tb = '|'.join(format_tb(err.__traceback__)).replace('\n', ';')
            msg = f"Failed to get HTR data for {output_path} with start time {min_time} and end time {max_time}: {err} ({type(err)}; {tb})"  # noqa: E501
            raise Level1AException(msg)

    try:
        dataframes = [rac_df, reconstructed_df, matched_schedule]
        if htr_bucket is not None:
            dataframes.append(htr_subset)
        merged = concat(dataframes, axis=1)
        if data_prefix == "CCD":
            add_ccd_item_attributes(merged)
        for key, val in metadata.items():
            merged[
                key if isinstance(key, str) else key.decode()
            ] = val if isinstance(val, str) else val.decode()
        out_table = pa.Table.from_pandas(merged)
        out_table = out_table.replace_schema_metadata({
            **metadata,
        })

        pq.write_table(
            out_table,
            f"{output_bucket}/{output_path}",
            filesystem=s3,
            version='2.6',
        )
    except Exception as err:
        tb = '|'.join(format_tb(err.__traceback__)).replace('\n', ';')
        msg = f"Failed to store {output_path} with start time {min_time} and end time {max_time}: {err} ({type(err)}; {tb})"  # noqa: E501
        raise Level1AException(msg)
