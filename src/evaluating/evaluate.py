import json
import os
import numpy as np
from transformers import Trainer
from utils.exceptions import EvaluationError


def save_confidence_reference(trainer: Trainer, output_path: str) -> None:
    """
        Computes confidence scores on the eval dataset and saves
        summary statistics to a JSON file for use as drift detection reference.
        Params:
            trainer (Trainer): The Trainer object after fine-tuning.
            output_path (str): Path where the reference JSON will be saved.
        Raises:
            EvaluationError: If there is an error during computation or saving.
    """
    try:
        predictions = trainer.predict(trainer.eval_dataset)
        logits = predictions.predictions
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        confidence = probs.max(axis=-1).tolist()

        stats = {
            "mean": float(np.mean(confidence)),
            "std": float(np.std(confidence)),
            "p25": float(np.percentile(confidence, 25)),
            "p50": float(np.percentile(confidence, 50)),
            "p75": float(np.percentile(confidence, 75)),
            "p95": float(np.percentile(confidence, 95)),
            "n_samples": len(confidence),
        }

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(
            f"Confidence reference saved to {output_path}: "
            f"mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
            f"n={stats['n_samples']}"
        )
    except Exception as e:
        raise EvaluationError(f"Error saving confidence reference: {e}")


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

        per_class = {
            k.removeprefix("eval_"): v
            for k, v in eval_results.items()
            if k.startswith(("eval_precision_", "eval_recall_", "eval_f1_"))
            and k != "eval_f1_macro"
        }

        return is_ready_for_hf_hub, {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "loss": loss,
            **per_class,
        }
    except Exception as e:
        raise EvaluationError(
            f"Error during fine-tuned model evaluation: {e}"
        )
