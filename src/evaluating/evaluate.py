from transformers import Trainer
from utils.exceptions import EvaluationError


def evaluate_hf_fine_tuned_model(
    trainer: Trainer, quality_thresholds: dict
) -> tuple[bool, dict]:
    """
        Evaluates the performance of the Hugging Face fine-tuned model
        and checks if it meets the specified quality thresholds for
        accuracy and F1 score.
        Params:
            trainer (Trainer): The Trainer object after fine-tuning.
            quality_thresholds (dict): A dictionary containing the minimum
                thresholds for accuracy and F1 score to decide whether
                the model is ready to be pushed to the Hugging Face Hub.
        Returns:
            tuple: A tuple containing a boolean indicating whether the
                model is ready to be pushed to the Hugging Face Hub and
                a dictionary with the evaluation metrics.
        Raises:
            EvaluationError: If there is an error during the evaluation.
    """
    try:
        is_ready_for_hf_hub = False

        accuracy_threshold = quality_thresholds.get("accuracy_min", 0.7)
        f1_score_threshold = quality_thresholds.get("f1_min", 0.7)

        eval_results = trainer.evaluate()

        accuracy = eval_results.get("eval_accuracy", 0)
        f1_macro = eval_results.get("eval_f1_macro", 0)
        loss = eval_results.get("eval_loss", 0)

        if (accuracy >= accuracy_threshold) and (
            f1_macro >= f1_score_threshold
        ):
            is_ready_for_hf_hub = True

        return is_ready_for_hf_hub, {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "loss": loss,
        }
    except Exception as e:
        raise EvaluationError(
            f"Error during fine-tuned model evaluation: {e}"
        )
