from unittest.mock import patch
from utils.exceptions import PredictionError


def test_predict_success(client):
    mock_result = {"label": "positive", "score": 0.95}
    with patch("api.routers.prediction.make_prediction", return_value=mock_result):
        response = client.post("/predict", json={"text": "I love this product!"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "positive"
    assert data["score"] == 0.95


def test_predict_error(client):
    with patch("api.routers.prediction.make_prediction", side_effect=PredictionError("model failed")):
        response = client.post("/predict", json={"text": "test"})
    assert response.status_code == 500
    assert "model failed" in response.json()["detail"]


def test_predict_missing_body(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_text_too_long(client):
    response = client.post("/predict", json={"text": "x" * 513})
    assert response.status_code == 422
