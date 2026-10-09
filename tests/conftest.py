import pathlib
import sys

import pytest
import torch

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(False)
    yield


@pytest.fixture
def small_cfg():
    return dict(d_model=32, n_heads=4, rope_fraction=0.5, window=2, n_rounds=1)
