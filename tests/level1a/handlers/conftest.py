from pathlib import Path

import pytest  # type: ignore

from pandas import DataFrame, Timestamp  # type: ignore


@pytest.fixture
def data_dir() -> Path:
    return Path(__file__).parent / "files"


@pytest.fixture
def rac_dir(data_dir: Path) -> Path:
    return data_dir / "rac"


@pytest.fixture
def platform_dir(data_dir: Path) -> Path:
    return data_dir / "platform"


@pytest.fixture
def ccd_path(rac_dir: Path) -> Path:
    return rac_dir / "CCD" / "2022" / "12" / "21" / "23" / "MATS_OPS_Level0_VC1_APID100_20221221-132606_20221222-133247.parquet"  # noqa: E501


@pytest.fixture
def schedule_path(data_dir: Path) -> Path:
    return data_dir / "schedule"


@pytest.fixture
def htr_path(rac_dir: Path) -> Path:
    return rac_dir / "HTR"


@pytest.fixture
def reconstructed_path(platform_dir: Path) -> Path:
    return platform_dir / "ReconstructedData"


@pytest.fixture
def schedule() -> DataFrame:
    return DataFrame(
        {
            "schedule_start_date": [
                Timestamp('1978-03-08 22:30:00.0'),
                Timestamp('1978-03-15 22:30:00.0'),
                Timestamp('1978-03-22 22:30:00.0'),
                Timestamp('1978-03-29 22:30:00.0'),
                Timestamp('1978-04-05 22:30:00.0'),
                Timestamp('1978-04-12 22:30:00.0'),
                Timestamp('2010-10-10 09:00:00.0'),
                Timestamp('2010-10-10 10:00:00.0'),
            ],
            "schedule_end_date": [
                Timestamp('1978-03-08 23:00:00.0'),
                Timestamp('1978-03-15 23:00:00.0'),
                Timestamp('1978-03-22 23:00:00.0'),
                Timestamp('1978-03-29 23:00:00.0'),
                Timestamp('1978-04-05 23:00:00.0'),
                Timestamp('1978-04-12 23:00:00.0'),
                Timestamp('2010-10-10 11:00:00.0'),
                Timestamp('2010-10-10 12:00:00.0'),
            ],
            "Answer": [39, 40, 41, 42, 43, 44, -1, -2],
            "Fit": [1, 2, 3, 4, 5, 6, -1, -1],
        }
    )
