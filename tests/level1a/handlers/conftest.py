from pathlib import Path

import pyarrow.dataset as ds  # type: ignore
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
def rac_dataset(rac_dir):
    return ds.dataset(rac_dir)


@pytest.fixture
def platform_dataset(platform_dir):
    return ds.dataset(platform_dir)


@pytest.fixture
def output_dataset(data_dir):
    return ds.dataset(data_dir / "output")


@pytest.fixture
def empty_dataset(data_dir):
    return ds.dataset(data_dir / "empty")
