import pytest
from unittest.mock import MagicMock, patch
from predicting.make_prediction import create_sentiment_pipeline, predict
from utils.exceptions import PredictionError


def test_create_sentiment_pipeline_success():
    mock_pipeline = MagicMock()
    with patch(
        "predicting.make_prediction.pipeline", return_value=mock_pipeline
    ):
        result = create_sentiment_pipeline("test-model")
    assert result == mock_pipeline


def test_create_sentiment_pipeline_error():
    with patch(
        "predicting.make_prediction.pipeline",
        side_effect=Exception("network error")
    ):
        with pytest.raises(PredictionError, match="Error loading model"):
            create_sentiment_pipeline("test-model")


def test_predict_success():
    mock_pipeline = MagicMock(return_value=[
        {"label": "positive", "score": 0.9},
        {"label": "negative", "score": 0.1},
    ])
    result = predict("I love this!", mock_pipeline)
    assert result["label"] == "positive"
    assert result["score"] == 0.9


def test_predict_error():
    mock_pipeline = MagicMock(side_effect=Exception("inference failed"))
    with pytest.raises(PredictionError, match="Error during prediction"):
        predict("some text", mock_pipeline)
