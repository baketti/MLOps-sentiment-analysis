import os
import requests
from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task

MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR", "/opt/airflow/models")
HF_TOKEN = os.getenv("HF_TOKEN", "")
CONFIG_PATH = "/opt/airflow/config.yaml"

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="sentiment_retraining",
    default_args=default_args,
    description="Retraining pipeline for the sentiment analysis model",
    schedule_interval="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["sentiment", "retraining"],
) as dag:

    @task
    def download_dataset_task():
        from loading.load_dataset import KaggleDatasetLoader
        from utils.config import load_config

        config = load_config(CONFIG_PATH)
        dataset_config = config["kaggle_dataset"]

        loader = KaggleDatasetLoader(
            dataset_config["name"],
            dataset_config["file_path"],
        )
        X, y = loader.load_and_get_sentiment_analysis_dataset()

        print(f"Dataset loaded: {len(X)} samples")

    @task
    def fine_tune_task() -> dict:
        from transformers import (
            AutoTokenizer, AutoModelForSequenceClassification
        )
        from training.train_model import (
            get_train_test_datasets,
            tokenize_train_test_datasets,
            fine_tune_model,
        )
        from evaluating.evaluate import (
            evaluate_hf_fine_tuned_model, save_confidence_reference
        )
        from utils.config import load_config

        config = load_config(CONFIG_PATH)
        hf_model_config = config["hf_model"]

        tokenizer = AutoTokenizer.from_pretrained(hf_model_config["name"])
        model = AutoModelForSequenceClassification.from_pretrained(
            hf_model_config["name"],
            num_labels=hf_model_config["num_labels"],
            ignore_mismatched_sizes=True,
        )

        train_dataset, test_dataset = get_train_test_datasets(
            config["kaggle_dataset"]["name"],
            config["kaggle_dataset"]["file_path"],
        )
        train_dataset, test_dataset = tokenize_train_test_datasets(
            train_dataset, test_dataset,
            tokenizer, hf_model_config["label2id"],
        )

        label_names = [
            k for k, _ in sorted(
                hf_model_config["label2id"].items(), key=lambda x: x[1]
            )
        ]
        trainer = fine_tune_model(
            train_dataset, test_dataset, model, tokenizer,
            MODEL_OUTPUT_DIR, config["hf_hub_model_id"], label_names,
        )

        trainer.save_model(MODEL_OUTPUT_DIR)
        tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

        _, metrics = evaluate_hf_fine_tuned_model(
            trainer, config["quality_thresholds"]
        )

        reference_path = os.path.join(
            MODEL_OUTPUT_DIR, "reference_confidence.json"
        )
        save_confidence_reference(trainer, reference_path)

        print(f"Training completed: {metrics}")
        return metrics

    @task
    def update_metrics_task(metrics: dict):
        requests.post(
            "http://fastapi:8000/metrics/training",
            json=metrics,
            timeout=10,
        )

    @task.short_circuit
    def quality_gate_task(metrics: dict) -> bool:
        from utils.config import load_config

        config = load_config(CONFIG_PATH)
        thresholds = config["quality_thresholds"]

        f1 = metrics.get("f1_macro", 0)
        accuracy = metrics.get("accuracy", 0)

        passed = (
            f1 >= thresholds["f1_min"]
            and accuracy >= thresholds["accuracy_min"]
        )

        if passed:
            print(f"Quality gate passed: F1={f1:.3f}, Accuracy={accuracy:.3f}")
        else:
            print(f"Quality gate failed: F1={f1:.3f}, Accuracy={accuracy:.3f}")

        return passed

    @task.short_circuit
    def drift_detection_task() -> bool:
        import json
        import numpy as np
        from scipy import stats

        PROMETHEUS_URL = "http://prometheus:9090"
        FASTAPI_URL = "http://fastapi:8000"
        REFERENCE_PATH = os.path.join(
            MODEL_OUTPUT_DIR, "reference_confidence.json"
        )
        DRIFT_THRESHOLD = 0.05

        if not os.path.exists(REFERENCE_PATH):
            print("No confidence reference found, skipping drift detection.")
            return False

        with open(REFERENCE_PATH) as f:
            reference = json.load(f)

        query = (
            "sum(rate(sentiment_prediction_confidence_bucket[24h])) by (le)"
        )
        response = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
            timeout=10,
        )
        response.raise_for_status()
        result = response.json().get("data", {}).get("result", [])

        if not result:
            print("No prediction confidence data in Prometheus, skipping.")
            return False

        buckets = sorted(
            [
                (float(r["metric"]["le"]), float(r["value"][1]))
                for r in result
            ],
            key=lambda x: x[0],
        )
        total = buckets[-1][1]
        if total == 0:
            print("No predictions recorded yet, skipping drift detection.")
            return False

        production_samples = np.array([
            le for le, count in buckets
            if le != float("inf")
            for _ in range(
                int(round((count / total) * reference["n_samples"]))
            )
        ])
        reference_samples = np.random.normal(
            loc=reference["mean"],
            scale=max(reference["std"], 1e-6),
            size=reference["n_samples"],
        )
        reference_samples = np.clip(reference_samples, 0, 1)

        if len(production_samples) == 0:
            print("Could not reconstruct production samples, skipping.")
            return False

        ks_stat, p_value = stats.ks_2samp(
            reference_samples, production_samples
        )
        drift_score = float(ks_stat)

        requests.post(
            f"{FASTAPI_URL}/metrics/drift",
            json={"score": drift_score},
            timeout=10,
        )

        drift_detected = p_value < DRIFT_THRESHOLD
        print(
            f"Drift detection: KS={ks_stat:.4f}, p-value={p_value:.4f}, "
            f"drift={'YES' if drift_detected else 'NO'}"
        )
        return drift_detected

    @task
    def push_to_hub_task():
        from transformers import (
            AutoModelForSequenceClassification, AutoTokenizer
        )
        from utils.config import load_config

        config = load_config(CONFIG_PATH)
        hub_model_id = config["hf_hub_model_id"]

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_OUTPUT_DIR
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_OUTPUT_DIR)

        model.push_to_hub(hub_model_id, token=HF_TOKEN)
        tokenizer.push_to_hub(hub_model_id, token=HF_TOKEN)
        print(f"Model pushed to HF Hub: {hub_model_id}")

    download_dataset = download_dataset_task()
    drift_check = drift_detection_task()
    fine_tune = fine_tune_task()
    update_metrics = update_metrics_task(fine_tune)
    quality_gate = quality_gate_task(fine_tune)

    drift_check >> download_dataset >> fine_tune
    update_metrics >> quality_gate >> push_to_hub_task()
