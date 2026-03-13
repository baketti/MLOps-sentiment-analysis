# Sentimenti Analysis MLOps

End-to-end MLOps pipeline for sentiment analysis on tweets using a fine-tuned RoBERTa model.

## Stack

| Service | Description |
|---|---|
| **Transformers** | Fine-tuning and inference with `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| **MLflow** | Experiment tracking, model registry, and artifact storage |
| **FastAPI** | REST API to serve real-time sentiment predictions |
| **Apache Airflow** | DAG-based orchestration of training and retraining pipelines |
| **Docker** | Containerisation of all services for reproducible environments |
| **Grafana** | Dashboard for monitoring model performance and API metrics |
| **Prometheus** | Metrics collection and alerting for the inference service |
| **Hugging Face Hub** | Remote model storage and versioning |
| **pytest** | Unit and integration testing of preprocessing, training, and API logic |
