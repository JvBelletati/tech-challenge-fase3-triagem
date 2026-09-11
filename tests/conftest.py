from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_csv() -> Path:
    """Small offline slice of the real dataset. Tests never hit the network."""
    return FIXTURES / "sample.csv"
