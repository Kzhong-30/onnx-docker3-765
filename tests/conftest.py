import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def model_path():
    return "my_model_quantized.onnx"


@pytest.fixture(scope="session")
def test_data_dir():
    return "hymenoptera_data/val"


@pytest.fixture(scope="session")
def classes():
    return ['ant', 'bee']
