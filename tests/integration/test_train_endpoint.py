from unittest.mock import patch
from utils.exceptions import FineTuningError


def test_train_success(client):
    mock_metrics = {"f1_macro": 0.82, "accuracy": 0.85, "loss": 0.67}
    with patch(
        "api.routers.training.train_and_save_model",
        return_value=mock_metrics
    ):
        response = client.post("/train")
    assert response.status_code == 200
    assert response.json()["status"] == "Training completed"


def test_train_error(client):
    with patch(
        "api.routers.training.train_and_save_model",
        side_effect=FineTuningError("GPU out of memory")
    ):
        response = client.post("/train")
    assert response.status_code == 500
    assert "GPU out of memory" in response.json()["detail"]
