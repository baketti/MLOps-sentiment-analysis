import pytest
import tempfile
import os
from utils.config import load_config
from utils.exceptions import ConfigLoadError


def test_load_config_valid():
    content = "hf_model:\n  name: test-model\n"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(content)
        path = f.name
    try:
        config = load_config(path)
        assert config["hf_model"]["name"] == "test-model"
    finally:
        os.unlink(path)


def test_load_config_file_not_found():
    with pytest.raises(ConfigLoadError):
        load_config("non_existent_file.yaml")


def test_load_config_invalid_yaml():
    content = "key: [\n  unclosed bracket"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False
    ) as f:
        f.write(content)
        path = f.name
    try:
        with pytest.raises(ConfigLoadError):
            load_config(path)
    finally:
        os.unlink(path)
