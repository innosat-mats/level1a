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
    return rac_dir / "CCD" / "2022" / "12" / "21" / "23" / "MATS_OPS_Level0_VC1_APID100_20221221-132606_20221222-133247.parquet"  # noqa: E501


@pytest.fixture
def htr_path(rac_dir):
    return rac_dir / "HTR"


@pytest.fixture
def reconstructed_path(platform_dir):
    return platform_dir / "ReconstructedData"
