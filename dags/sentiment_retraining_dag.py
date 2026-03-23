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
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from training.train_model import (
            get_train_test_datasets,
            tokenize_train_test_datasets,
            fine_tune_model,
        )
        from evaluating.evaluate import evaluate_hf_fine_tuned_model
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
            train_dataset, test_dataset, tokenizer, hf_model_config["label2id"],
        )

        trainer = fine_tune_model(
            train_dataset, test_dataset, model, tokenizer,
            MODEL_OUTPUT_DIR, config["hf_hub_model_id"],
        )

        trainer.save_model(MODEL_OUTPUT_DIR)
        tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

        _, metrics = evaluate_hf_fine_tuned_model(
            trainer, config["quality_thresholds"]
        )

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

    @task
    def push_to_hub_task():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from utils.config import load_config

        config = load_config(CONFIG_PATH)
        hub_model_id = config["hf_hub_model_id"]

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_OUTPUT_DIR)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_OUTPUT_DIR)

        model.push_to_hub(hub_model_id, token=HF_TOKEN)
        tokenizer.push_to_hub(hub_model_id, token=HF_TOKEN)
        print(f"Model pushed to HF Hub: {hub_model_id}")

    download_dataset = download_dataset_task()
    fine_tune = fine_tune_task()
    update_metrics = update_metrics_task(fine_tune)
    quality_gate = quality_gate_task(fine_tune)

    download_dataset >> fine_tune
    update_metrics >> quality_gate >> push_to_hub_task()
