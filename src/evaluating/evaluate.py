from sklearn.metrics import f1_score, accuracy_score
from transformers import Trainer
from utils.exceptions import EvaluationError


def evaluate_hf_pretrained_model(pipeline, test_dataset, label2id):
    """
        Evaluates the performance of the Hugging Face pretrained model on the test dataset using F1 macro and accuracy metrics.
        Params:
            pipeline: The Hugging Face pipeline for text classification.
            test_dataset: The test dataset as a Hugging Face Dataset.
            label2id: A dictionary mapping label names to their corresponding IDs.
        Returns:
            dict: A dictionary containing the F1 macro score and accuracy of the model on the test dataset.
    """
    try:
        y_true, y_pred = [], []

        for data in test_dataset:
            result = pipeline(data["text"])
            best   = max(result, key=lambda x: x["score"])
            y_true.append(data["label"])
            y_pred.append(label2id[best["label"].lower()])

        return {
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "accuracy": accuracy_score(y_true, y_pred)
        }
    except Exception as e:
        raise EvaluationError(f"Error during pretrained model evaluation: {e}")


def evaluate_hf_fine_tuned_model(trainer: Trainer, quality_thresholds: dict) -> tuple[bool, dict]:
    """
        Evaluates the performance of the Hugging Face fine-tuned model using and checks if it meets the specified quality thresholds for accuracy and F1 score.
        Params:
            trainer (Trainer): The Trainer object after fine-tuning the model.
            quality_thresholds (dict): A dictionary containing the minimum thresholds for accuracy and F1 score to decide whether the model is ready to be pushed to the Hugging Face Hub.
        Returns:
            tuple: A tuple containing a boolean indicating whether the model is ready to be pushed to the Hugging Face Hub and a dictionary with the evaluation metrics (accuracy and F1 macro score).
    """
    try:
        is_ready_for_hf_hub = False

        accuracy_threshold = quality_thresholds.get("accuracy_min", 0.7)
        f1_score_threshold  = quality_thresholds.get("f1_min", 0.7)

        eval_results = trainer.evaluate()

        accuracy = eval_results.get("eval_accuracy", 0)
        f1_macro = eval_results.get("eval_f1_macro", 0)

        if (accuracy >= accuracy_threshold) and (f1_macro >= f1_score_threshold):
            is_ready_for_hf_hub = True

        return is_ready_for_hf_hub, {
            "accuracy": accuracy,
            "f1_macro": f1_macro
        }
    except Exception as e:
        raise EvaluationError(f"Error during fine-tuned model evaluation: {e}")