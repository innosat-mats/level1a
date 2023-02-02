import json
import os
from http import HTTPStatus
from typing import Any, Dict, Optional, Tuple

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
)

Event = Dict[str, Any]
Context = Any

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
    ("hour", pa.int8()),
])

PLATFORM_PARTITIONS = pa.schema([
    ("year", pa.int16()),
    ("month", pa.int8()),
    ("day", pa.int8()),
])


class DoesNotCover(Exception):
    pass


class InvalidMessage(Exception):
    pass


class Level1AException(Exception):
    pass


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


def get_ccd_records(
    path_or_bucket: str,
    filesystem: pa.fs.FileSystem = None,
) -> Tuple[DataFrame, pq.FileMetaData]:
    table = pq.read_table(
        path_or_bucket,
        filesystem=filesystem,
    )
    return (
        table.to_pandas().set_index("EXPDate").sort_index(),
        table.schema.metadata,
    )


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
                    & (ds.field('day') >= max_time.day)
                )
            )
            & (
                (
                    (ds.field('hour') >= min_time.hour)
                    & (ds.field('hour') <= max_time.hour)
                ) | (
                    (ds.field('day') == min_time.day)
                    & (ds.field('hour') >= min_time.hour)
                ) | (
                    (ds.field('day') == max_time.day)
                    & (ds.field('hour') >= max_time.hour)
                )
            )
            & (ds.field('TMHeaderTime') >= min_time)
            & (ds.field('TMHeaderTime') <= max_time)
        ),
        columns=HTR_COLUMNS,
    ).to_pandas().drop_duplicates("TMHeaderTime").set_index(
        "TMHeaderTime"
    ).sort_index()
    return dataset


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
                & (ds.field('day') >= max_time.day)
            )
        )
        & (ds.field('time') >= min_time.asm8)
        & (ds.field('time') <= max_time.asm8)
    )).to_pandas().drop_duplicates("time").set_index("time").sort_index()
    dataset.index = dataset.index.tz_localize('utc')
    dataset.drop(columns=["year", "month", "day"], inplace=True)
    return dataset


def get_search_bounds(
    timeinds: DatetimeIndex
) -> Tuple[Timestamp, Timestamp]:
    return timeinds.min(), timeinds.max()


def get_offset(frequency: float) -> Timedelta:
    return OFFSET_FACTOR*Timedelta(seconds=1/frequency)


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


def lambda_handler(event: Event, context: Context):
    try:
        output_bucket = get_or_raise("OUTPUT_BUCKET")
        platform_bucket = get_or_raise("PLATFORM_BUCKET")
        htr_bucket = get_or_raise("HTR_BUCKET")
        version = get_or_raise("L1A_VERSION")
        region = os.environ.get('AWS_REGION', "eu-north-1")
        s3 = pa.fs.S3FileSystem(region=region)

        try:
            bucket, object = parse_event_message(event)
        except InvalidMessage:
            return {
                'statusCode': HTTPStatus.NO_CONTENT,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'message': 'Failed to parse event, nothing to do.'
                })
            }

        if not object.endswith(".parquet"):
            return {
                'statusCode': HTTPStatus.NO_CONTENT,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({
                    'message': f'{object} is not a parquet file, nothing to do.'
                })
            }

        rac_df, metadata = get_ccd_records(
            f"{bucket}/{object}",
            filesystem=s3,
        )
        metadata.update({"L1ACode": version})

        min_time, max_time = get_search_bounds(rac_df.index)
    except Exception as err:
        raise Level1AException(f"Failed to initialize handler: {err}") from err

    try:
        reconstructed_df = get_reconstructed_records(
            f"{platform_bucket}/ReconstructedData",
            min_time - get_offset(RECONSTRUCTED_FREQUENCY),
            max_time + get_offset(RECONSTRUCTED_FREQUENCY),
            filesystem=s3,
        )
        htr_df = get_htr_records(
            f"{htr_bucket}/HTR",
            min_time - get_offset(HTR_FREQUENCY),
            max_time + get_offset(HTR_FREQUENCY),
            filesystem=s3,
        )

        if not covers(reconstructed_df.index, min_time, max_time):
            raise DoesNotCover("Reconstructed data is missing timestamps")

        reconstructed_df = interpolate(
            reconstructed_df,
            rac_df.index,
            max_diff=get_offset(RECONSTRUCTED_FREQUENCY),
        )
        htr_subset = interpolate(
            htr_df,
            rac_df.index,
            max_diff=get_offset(HTR_FREQUENCY),
        )
        out_table = pa.Table.from_pandas(concat(
            [rac_df, reconstructed_df, htr_subset],
            axis=1,
        ))
        out_table.replace_schema_metadata({
            **out_table.schema.metadata,
            **metadata,
        })

        pq.write_table(
            out_table,
            f"{output_bucket}/{object.strip('/CCD')}",
            filesystem=s3,
            version='2.6',
        )
    except Exception as err:
        msg = f"Failed to process {object} with start time {min_time} and end time {max_time}: {err}"  # noqa: E501
        raise Level1AException(msg) from err
