import copy
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

FAKE_CONFIG = {
    "hf_model": {
        "name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "num_labels": 3,
        "label2id": {"negative": 0, "neutral": 1, "positive": 2},
    },
    "hf_hub_model_id": "test-user/test-finetuned-model",
    "quality_thresholds": {"f1_min": 0.7, "accuracy_min": 0.7},
    "kaggle_dataset": {"name": "test/dataset", "file_path": "test.csv"},
}


@pytest.fixture
def client():
    with patch("api.main.load_config", side_effect=lambda: copy.deepcopy(FAKE_CONFIG)), \
         patch("api.main.AutoTokenizer.from_pretrained", return_value=MagicMock()), \
         patch("api.main.AutoModelForSequenceClassification.from_pretrained", return_value=MagicMock()):
        from api.main import app
        with TestClient(app) as c:
            yield c
