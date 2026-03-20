import pytest
from unittest.mock import MagicMock
from evaluating.evaluate import evaluate_hf_fine_tuned_model
from utils.exceptions import EvaluationError


def test_evaluate_passes_thresholds():
    mock_trainer = MagicMock()
    mock_trainer.evaluate.return_value = {"eval_accuracy": 0.85, "eval_f1_macro": 0.80}
    is_ready, metrics = evaluate_hf_fine_tuned_model(
        mock_trainer, {"accuracy_min": 0.7, "f1_min": 0.7}
    )
    assert is_ready is True
    assert metrics["accuracy"] == 0.85
    assert metrics["f1_macro"] == 0.80


def test_evaluate_fails_thresholds():
    mock_trainer = MagicMock()
    mock_trainer.evaluate.return_value = {"eval_accuracy": 0.5, "eval_f1_macro": 0.4}
    is_ready, _ = evaluate_hf_fine_tuned_model(
        mock_trainer, {"accuracy_min": 0.7, "f1_min": 0.7}
    )
    assert is_ready is False


def test_evaluate_error():
    mock_trainer = MagicMock()
    mock_trainer.evaluate.side_effect = Exception("evaluation failed")
    with pytest.raises(EvaluationError, match="Error during fine-tuned model evaluation"):
        evaluate_hf_fine_tuned_model(mock_trainer, {"accuracy_min": 0.7, "f1_min": 0.7})
