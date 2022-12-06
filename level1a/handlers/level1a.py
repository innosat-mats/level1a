import os
from datetime import timezone
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

last_date: Optional[Timestamp] = None
DEFAULT_START = Timestamp(2022, 11, 20, tzinfo=timezone.utc)
OFFSET_SECONDS = 30
LOOKBACK_DAYS = 3
RAC_PREFIXES = {"CCD", "CPRU", "HTR", "PM", "PWR", "STAT", "TCV"}
PLATFORM_PREFIXES = {
    "HK_ecPowOps_1", "PreciseAttitudeEstimation", "PreciseOrbitEstimation",
    "scoCurrentScMode", "TM_acGnssOps", "TM_afAcsHiRateAttitudeData",
}


def get_or_raise(variable_name: str) -> str:
    if (var := os.environ.get(variable_name)) is None:
        raise EnvironmentError(
            f"{variable_name} is a required environment variable"
        )
    return var


def get_last_date(
    dataset: ds.FileSystemDataset,
    start: Timestamp,
) -> Optional[Timestamp]:
    try:
        dates = dataset.to_table(
            filter=ds.field('EXPDate') >= start, columns=['EXPDate']
        )['EXPDate'].to_pandas()
    except pa.ArrowInvalid:
        return None
    if len(dates) == 0:
        return None
    return dates.max()


def get_ccd_records(
    path_or_bucket: str,
    min_time: Timestamp,
    filesystem: pa.fs.FileSystem = None,
) -> DataFrame:
    return ds.dataset(
        path_or_bucket,
        filesystem=filesystem,
        ignore_prefixes=list(RAC_PREFIXES - {"CCD"}),
    ).to_table(
        filter=ds.field('EXPDate') >= min_time
    ).to_pandas().set_index("EXPDate").sort_index()


def get_orbit_records(
    path_or_bucket: str,
    min_time: np.datetime64,
    max_time: np.datetime64,
    filesystem: pa.fs.FileSystem = None,
) -> DataFrame:
    dataset = ds.dataset(
        path_or_bucket,
        filesystem=filesystem,
        ignore_prefixes=list(PLATFORM_PREFIXES - {"PreciseOrbitEstimation"}),
        schema=pa.schema([
            ("time", pa.timestamp('ns')),
            ("afsTangentPoint", pa.list_(pa.float64())),
            ("acsGnssStateJ2000", pa.list_(pa.float64())),
        ])
    ).to_table(
        filter=(ds.field('time') >= min_time) & (ds.field('time') <= max_time)
    ).to_pandas().drop_duplicates("time").set_index("time").sort_index()
    dataset.index = dataset.index.tz_localize('utc')
    return dataset


def get_attitude_records(
    path_or_bucket: str,
    min_time: np.datetime64,
    max_time: np.datetime64,
    filesystem: pa.fs.FileSystem = None,
) -> DataFrame:
    dataset = ds.dataset(
        path_or_bucket,
        filesystem=filesystem,
        ignore_prefixes=list(PLATFORM_PREFIXES - {"PreciseAttitudeEstimation"}),
        schema=pa.schema([
            ("time", pa.timestamp('ns')),
            ("afsAttitudeState", pa.list_(pa.float64())),
        ])
    ).to_table(
        filter=(ds.field('time') >= min_time) & (ds.field('time') <= max_time)
    ).to_pandas().drop_duplicates("time").set_index("time").sort_index()
    dataset.index = dataset.index.tz_localize('utc')
    return dataset


def get_search_bounds(
    timeinds: DatetimeIndex
) -> Tuple[np.datetime64, np.datetime64]:
    return (
        timeinds.min().asm8 - np.timedelta64(OFFSET_SECONDS, 's'),
        timeinds.max().asm8 + np.timedelta64(OFFSET_SECONDS, 's')
    )


def interp_array(
    eval_ind: float,
    indices: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    diffs = np.abs(eval_ind - indices)
    if np.min(diffs) == 0:
        return values[np.argmin(diffs)]
    dt = np.diff(indices)
    weights = diffs[::-1]/dt
    return np.average(values, weights=weights, axis=0)


def interp_to(
    target_date: Timestamp,
    column: Series,
) -> np.ndarray:
    before = column.index.get_indexer([target_date], method='ffill')[0]
    after = column.index.get_indexer([target_date], method='bfill')[0]

    timestamps = column.index[[before, after]].astype(int).values
    return interp_array(
        target_date.value,
        timestamps,
        column.iloc[[before, after]].values,
    )


def interpolate(dataframe: DataFrame, target: DatetimeIndex) -> DataFrame:
    return DataFrame({
        column: [interp_to(ind, dataframe[column]) for ind in target]
        for column in dataframe
    }, index=target)


def get_partitioned_dates(datetimes: DatetimeIndex) -> DataFrame:
    return DataFrame({
        'year': datetimes.year,
        'month': datetimes.month,
        'day': datetimes.day,
    }, index=datetimes)


def get_filename(timeinds: DatetimeIndex) -> str:
    return "".join([
        "payload-level1a_",
        timeinds.min().strftime('%Y%m%d-%H%M%S'),
        "_",
        timeinds.max().strftime('%Y%m%d-%H%M%S'),
        "_{i}.parquet"
    ])


def lambda_handler(event: Event, context: Context):
    global last_date
    output_bucket = get_or_raise("OUTPUT_BUCKET")
    rac_bucket = get_or_raise("RAC_BUCKET")
    platform_bucket = get_or_raise("PLATFORM_BUCKET")
    region = os.environ.get('AWS_REGION', "eu-north-1")
    s3 = pa.fs.S3FileSystem(region=region)

    target_dataset = ds.dataset(output_bucket, filesystem=s3)

    last_date = (
        last_date
        or get_last_date(
            target_dataset,
            Timestamp.now(tz=timezone.utc) - Timedelta(days=LOOKBACK_DAYS),
        )
        or get_last_date(
            target_dataset,
            DEFAULT_START,
        )
        or DEFAULT_START
    )

    rac_df = get_ccd_records(
        rac_bucket,
        last_date,
        filesystem=s3,
    )

    min_time, max_time = get_search_bounds(rac_df.index)

    attitude_df = get_attitude_records(
        platform_bucket,
        min_time,
        max_time,
        filesystem=s3,
    )
    orbit_df = get_orbit_records(
        platform_bucket,
        min_time,
        max_time,
        filesystem=s3,
    )
    attitude_subset = interpolate(attitude_df, rac_df.index)
    orbit_subset = interpolate(orbit_df, rac_df.index)
    partitioned_dates = get_partitioned_dates(rac_df.index)
    out_table = pa.Table.from_pandas(concat(
        [rac_df, attitude_subset, orbit_subset, partitioned_dates],
        axis=1,
    ))

    pq.write_to_dataset(
        table=[out_table],
        root_path=output_bucket,
        basename_template=get_filename(rac_df.index),
        existing_data_behavior="overwrite_or_ignore",
        filesystem=s3,
        partitioning=ds.partitioning(
            schema=pa.schema([
                ('year', pa.int32()),
                ('month', pa.int32()),
                ('day', pa.int32()),
            ]),
        ),
        version='2.6',
    )

    last_date = rac_df.index.max()
