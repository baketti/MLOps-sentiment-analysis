import pytest
from unittest.mock import MagicMock, patch
from training.train_model import fine_tune_model, save_and_push_model_on_hf_hub
from utils.exceptions import FineTuningError, PushingToHubError


def test_fine_tune_model_success():
    mock_trainer = MagicMock()
    with patch("training.train_model.DataCollatorWithPadding"), \
         patch("training.train_model.TrainingArguments"), \
         patch("training.train_model.Trainer", return_value=mock_trainer):
        result = fine_tune_model(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(),
            "/tmp/out", "user/model", ["negative", "neutral", "positive"]
        )
    assert result == mock_trainer


def test_fine_tune_model_error():
    mock_trainer = MagicMock()
    mock_trainer.train.side_effect = Exception("CUDA OOM")
    with patch("training.train_model.DataCollatorWithPadding"), \
         patch("training.train_model.TrainingArguments"), \
         patch("training.train_model.Trainer", return_value=mock_trainer):
        with pytest.raises(FineTuningError, match="Error during fine-tuning"):
            fine_tune_model(
                MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                "/tmp/out", "user/model", ["negative", "neutral", "positive"]
            )


def test_save_and_push_passes_quality_gate():
    mock_trainer = MagicMock()
    with patch(
        "training.train_model.evaluate_hf_fine_tuned_model",
        return_value=(True, {"accuracy": 0.85, "f1_macro": 0.82})
    ):
        save_and_push_model_on_hf_hub(
            mock_trainer, MagicMock(), "/tmp/out",
            {"accuracy_min": 0.7, "f1_min": 0.7}
        )
    mock_trainer.push_to_hub.assert_called_once()


def test_save_and_push_fails_quality_gate():
    mock_trainer = MagicMock()
    with patch(
        "training.train_model.evaluate_hf_fine_tuned_model",
        return_value=(False, {"accuracy": 0.5, "f1_macro": 0.4})
    ):
        save_and_push_model_on_hf_hub(
            mock_trainer, MagicMock(), "/tmp/out",
            {"accuracy_min": 0.7, "f1_min": 0.7}
        )
    mock_trainer.push_to_hub.assert_not_called()


def test_save_and_push_hub_error():
    mock_trainer = MagicMock()
    mock_trainer.push_to_hub.side_effect = Exception("Hub unavailable")
    with patch(
        "training.train_model.evaluate_hf_fine_tuned_model",
        return_value=(True, {"accuracy": 0.85, "f1_macro": 0.82})
    ):
        with pytest.raises(PushingToHubError, match="Error pushing model"):
            save_and_push_model_on_hf_hub(
                mock_trainer, MagicMock(), "/tmp/out",
                {"accuracy_min": 0.7, "f1_min": 0.7}
            )
