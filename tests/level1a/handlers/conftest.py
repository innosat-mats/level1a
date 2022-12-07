from pathlib import Path

import pytest  # type: ignore


@pytest.fixture
def data_dir():
    return Path(__file__).parent / "files"


@pytest.fixture
def rac_dir(data_dir):
    return data_dir / "rac"


@pytest.fixture
def platform_dir(data_dir):
    return data_dir / "platform"


@pytest.fixture
def ccd_path(rac_dir):
    return rac_dir / "CCD" / "2022" / "11" / "22" / "MATS_OPS_Level0_VC1_APID100_20221122-080636_20221122-094142.parquet"  # noqa: E501
