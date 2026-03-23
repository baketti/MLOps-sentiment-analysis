# 🇮🇹 Sentiment Analysis MLOps

## Panoramica

Questo progetto è un **Proof of Concept (PoC)** di una pipeline MLOps multi-container per l'analisi del sentiment su testi in stile tweet. Dimostra come integrare fine-tuning di modelli, tracciamento degli esperimenti, servizio REST API e monitoraggio in un ambiente containerizzato e riproducibile.

Il sistema è costruito attorno a un modello RoBERTa fine-tuned ed espone due funzionalità principali tramite un'applicazione FastAPI: addestramento del modello su richiesta (fine-tuning) e predizione del sentiment in tempo reale. Tutti i servizi sono eseguiti come container Docker orchestrati tramite Docker Compose.

Il codice segue le convenzioni di stile **PEP 8** ed è validato ad ogni esecuzione della CI tramite `flake8`.

---

## Tecnologie

| Servizio | Descrizione |
|---|---|
| **Transformers** | Fine-tuning e inferenza con `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| **PyTorch (CPU)** | Backend di deep learning — build CPU-only per portabilità e riduzione della dimensione dell'immagine |
| **MLflow** | Tracciamento degli esperimenti, model registry e archiviazione degli artefatti |
| **FastAPI** | API REST per servire predizioni di sentiment in tempo reale e avviare il training |
| **Apache Airflow** | Orchestrazione tramite DAG dei pipeline di training e retraining |
| **Docker** | Containerizzazione di tutti i servizi per ambienti riproducibili |
| **Prometheus** | Raccolta delle metriche esposte dall'API (latenza di inferenza, predizioni per label, metriche di training) |
| **Grafana** | Dashboard di monitoraggio per la visualizzazione delle metriche raccolte da Prometheus |
| **Hugging Face Hub** | Storage remoto e versionamento del modello fine-tuned |
| **pytest** | Test unitari e di integrazione su caricamento dati, training e logica API |

---

## Funzionalità e Flusso Principale

### Endpoint API

| Metodo | Path | Descrizione |
|---|---|---|
| `POST` | `/train` | Avvia la pipeline completa di fine-tuning sul dataset Kaggle configurato |
| `POST` | `/predict` | Restituisce l'etichetta di sentiment e il punteggio di confidenza per un testo in input |
| `GET` | `/metrics` | Espone le metriche nel formato Prometheus per lo scraping |

### Flusso di training (`POST /train`)

1. Download del dataset da Kaggle tramite `kagglehub`
2. Suddivisione in train/test (80/20, stratificata)
3. Tokenizzazione con `AutoTokenizer` (`max_length=128`, padding dinamico)
4. Fine-tuning del modello con HuggingFace `Trainer`; le metriche vengono riportate automaticamente a MLflow
5. Valutazione del modello fine-tuned rispetto alle soglie di qualità definite in `config.yaml`:
   - `f1_min: 0.7`
   - `accuracy_min: 0.7`
6. Se entrambe le soglie sono superate, il modello viene pubblicato su Hugging Face Hub; altrimenti viene salvato solo in locale

### Flusso di predizione (`POST /predict`)

All'avvio dell'applicazione il **modello base** (`cardiffnlp/twitter-roberta-base-sentiment-latest`) viene caricato in memoria. Ad ogni richiesta di predizione il servizio verifica se il modello fine-tuned è disponibile su Hugging Face Hub:

- Se il modello fine-tuned è presente sull'Hub → viene utilizzato per l'inferenza
- Altrimenti → viene usato il modello base come fallback

La risposta contiene l'etichetta di sentiment predetta (`positive`, `neutral` o `negative`), il punteggio di confidenza e il nome del modello che ha prodotto la predizione.

---

## Risultati del Fine-tuning

Le seguenti metriche sono state ottenute dopo il fine-tuning sul dataset Kaggle:

| Metrica | Valore |
|---|---|
| **Loss** | 0.6712 |
| **Accuracy** | 0.87 |
| **F1 Macro** | 0.8673 |

Sia accuracy che F1 Macro superano le soglie di qualità configurate in `config.yaml`, quindi il modello è stato pubblicato automaticamente su Hugging Face Hub.

---

## Struttura del Progetto

```
sentiment-mlops/
├── src/
│   ├── main.py                        # Entry point Uvicorn
│   ├── api/
│   │   ├── main.py                    # App FastAPI e lifespan (modello caricato al boot)
│   │   ├── routers/                   # Handler degli endpoint (training, prediction, metrics)
│   │   ├── services/                  # Logica applicativa (orchestrazione training, inferenza)
│   │   ├── schemas/                   # Modelli Pydantic per request/response
│   │   └── utils/
│   │       ├── utilities.py           # resolve_model: selezione tra modello fine-tuned e base
│   │       └── metrics.py             # Definizioni metriche Prometheus (counter, histogram, gauge)
│   ├── loading/load_dataset.py        # KaggleDatasetLoader
│   ├── training/train_model.py        # Pipeline di fine-tuning
│   ├── evaluating/evaluate.py         # Valutazione e quality gate
│   ├── predicting/make_prediction.py  # Pipeline di inferenza
│   └── utils/
│       ├── config.py                  # Caricamento configurazione YAML
│       └── exceptions.py              # Gerarchia di eccezioni personalizzate
├── tests/
│   ├── unit/                          # Test unitari
│   └── integration/                   # Test di integrazione (FastAPI TestClient)
├── docker-env/
│   ├── Dockerfile                     # python:3.12-slim, PyTorch CPU-only
│   └── docker-compose.yml             # FastAPI (8000) + MLflow (5000) + Prometheus (9090) + Grafana (3000)
├── monitoring/
│   └── prometheus.yml                 # Configurazione scrape Prometheus
├── .github/workflows/
│   ├── ci.yml                         # Lint + test unitari + test integrazione ad ogni push
│   └── cd.yml                         # Continuous Delivery: build e push immagine Docker su DockerHub al merge su main
├── config.yaml                        # Nome modello, mapping etichette, soglie qualità, config dataset
├── requirements.txt                   # Dipendenze Python
├── pyproject.toml                     # Configurazione build del pacchetto (setuptools)
└── .env.example                       # Template variabili d'ambiente
```

---

## CI/CD

- **CI** (`ci.yml`): eseguita ad ogni push e sulle pull request verso `main`. Passi: checkout del codice, setup Python 3.12, installazione dipendenze, lint con `flake8`, test unitari, test di integrazione.
- **CD** (`cd.yml`): implementa la **Continuous Delivery** (non il Deployment). Si attiva automaticamente dopo una CI riuscita su `main`, costruisce l'immagine Docker e la pubblica su DockerHub con il tag `latest` e un tag con lo SHA del commit. Il deployment effettivo su un ambiente target rimane un passaggio manuale.

---

## Configurazione

I parametri principali sono centralizzati in `config.yaml`:

```yaml
hf_model:
  name: cardiffnlp/twitter-roberta-base-sentiment-latest
  num_labels: 3

hf_hub_model_id: Emanueleb/twitter-roberta-base-sentiment-latest-finetuned

quality_thresholds:
  f1_min: 0.7
  accuracy_min: 0.7

kaggle_dataset:
  name: mdismielhossenabir/sentiment-analysis
  file_path: sentiment_analysis.csv
```

Variabili d'ambiente (vedi `.env.example`):

```
HF_TOKEN=<huggingface_token>
MODEL_OUTPUT_DIR=<percorso_locale_output_modello>
MLFLOW_EXPERIMENT_NAME=<nome_esperimento>
HOST=127.0.0.1
PORT=3000
```

---

## Avvio Rapido

```bash
# Clona il repository
git clone <repo-url>
cd sentiment-mlops

# Copia e compila le variabili d'ambiente
cp .env.example .env

# Avvia tutti i servizi
cd docker-env
docker compose up --build
```

Servizi disponibili:
- FastAPI: `http://localhost:8000`
- MLflow UI: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (credenziali di default: `admin` / `admin`)

---

## Miglioramenti Futuri

- **Caricamento dataset personalizzato**: consentire agli utenti di fornire un proprio dataset al momento del training per abilitare il retraining su dati specifici del dominio senza modificare il codice.
- **Qualità del codice**: introdurre design pattern (es. Repository, Strategy, Factory) per rendere il codice più manutenibile e scalabile all'aumentare dei modelli e delle sorgenti dati supportati.
- **Selezione del modello per l'inferenza**: esporre un meccanismo per scegliere quale versione del modello utilizzare per la predizione — il modello base, l'ultimo modello fine-tuned o una versione specifica dalla cronologia dei fine-tuning memorizzata su Hugging Face Hub.

---

---

# 🇬🇧 Sentiment Analysis MLOps

## Overview

This project is a **Proof of Concept (PoC)** of a multi-container MLOps pipeline for sentiment analysis on tweet-style text. It demonstrates how to bring together model fine-tuning, experiment tracking, REST API serving, and monitoring in a reproducible, containerised environment.

The system is built around a fine-tuned RoBERTa model and exposes two main capabilities via a FastAPI application: on-demand model training (fine-tuning) and real-time sentiment prediction. All services run as Docker containers orchestrated through Docker Compose.

The codebase follows **PEP 8** style conventions and is validated at every CI run via `flake8`.

---

## Technologies

| Service | Description |
|---|---|
| **Transformers** | Fine-tuning and inference with `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| **PyTorch (CPU)** | Deep learning backend — CPU-only build used for portability and reduced image size |
| **MLflow** | Experiment tracking, model registry, and artifact storage |
| **FastAPI** | REST API to serve real-time sentiment predictions and trigger training |
| **Apache Airflow** | DAG-based orchestration of training and retraining pipelines |
| **Docker** | Containerisation of all services for reproducible environments |
| **Prometheus** | Scrapes metrics exposed by the API (inference latency, predictions by label, training metrics) |
| **Grafana** | Monitoring dashboard for visualising metrics collected by Prometheus |
| **Hugging Face Hub** | Remote model storage and versioning of the fine-tuned model |
| **pytest** | Unit and integration testing of data loading, training, and API logic |

---

## Features and Main Flow

### API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/train` | Triggers the full fine-tuning pipeline on the configured Kaggle dataset |
| `POST` | `/predict` | Returns the sentiment label and confidence score for an input text |
| `GET` | `/metrics` | Exposes metrics in Prometheus format for scraping |

### Training flow (`POST /train`)

1. Download the dataset from Kaggle via `kagglehub`
2. Split into train/test sets (80/20, stratified)
3. Tokenize with `AutoTokenizer` (`max_length=128`, dynamic padding)
4. Fine-tune the model using HuggingFace `Trainer`; metrics are reported to MLflow automatically
5. Evaluate the fine-tuned model against the quality thresholds defined in `config.yaml`:
   - `f1_min: 0.7`
   - `accuracy_min: 0.7`
6. If both thresholds are met, the model is pushed to Hugging Face Hub; otherwise it is saved locally only

### Prediction flow (`POST /predict`)

At application startup the **base model** (`cardiffnlp/twitter-roberta-base-sentiment-latest`) is loaded into memory. On every prediction request the service checks whether the fine-tuned model is available on Hugging Face Hub:

- If the fine-tuned model exists on the Hub → it is used for inference
- Otherwise → the base model is used as fallback

The response contains the predicted sentiment label (`positive`, `neutral`, or `negative`), the confidence score, and the name of the model that produced the prediction.

---

## Fine-tuning Results

The following metrics were obtained after fine-tuning on the Kaggle sentiment dataset:

| Metric | Value |
|---|---|
| **Loss** | 0.6712 |
| **Accuracy** | 0.87 |
| **F1 Macro** | 0.8673 |

Both accuracy and F1 Macro exceed the quality gate thresholds configured in `config.yaml`, so the model was automatically pushed to Hugging Face Hub.

---

## Project Structure

```
sentiment-mlops/
├── src/
│   ├── main.py                        # Uvicorn entry point
│   ├── api/
│   │   ├── main.py                    # FastAPI app and lifespan (model loaded at boot)
│   │   ├── routers/                   # Route handlers (training, prediction, metrics)
│   │   ├── services/                  # Business logic (training orchestration, inference)
│   │   ├── schemas/                   # Pydantic request/response models
│   │   └── utils/
│   │       ├── utilities.py           # resolve_model: fine-tuned vs base model selection
│   │       └── metrics.py             # Prometheus metric definitions (counter, histogram, gauge)
│   ├── loading/load_dataset.py        # KaggleDatasetLoader
│   ├── training/train_model.py        # Fine-tuning pipeline
│   ├── evaluating/evaluate.py         # Quality gate evaluation
│   ├── predicting/make_prediction.py  # Inference pipeline
│   └── utils/
│       ├── config.py                  # YAML configuration loader
│       └── exceptions.py              # Custom exception hierarchy
├── tests/
│   ├── unit/                          # Unit tests
│   └── integration/                   # Integration tests (FastAPI TestClient)
├── docker-env/
│   ├── Dockerfile                     # python:3.12-slim, CPU-only PyTorch
│   └── docker-compose.yml             # FastAPI (8000) + MLflow (5000) + Prometheus (9090) + Grafana (3000)
├── monitoring/
│   └── prometheus.yml                 # Prometheus scrape configuration
├── .github/workflows/
│   ├── ci.yml                         # Lint + unit + integration tests on every push
│   └── cd.yml                         # Continuous Delivery: builds and pushes Docker image to DockerHub on main
├── config.yaml                        # Model name, label mapping, quality thresholds, dataset config
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Package build config (setuptools)
└── .env.example                       # Environment variable template
```

---

## CI/CD

- **CI** (`ci.yml`): runs on every push and on pull requests to `main`. Steps: code checkout, Python 3.12 setup, dependency installation, `flake8` lint, unit tests, integration tests.
- **CD** (`cd.yml`): implements **Continuous Delivery** (not Deployment). It triggers automatically after a successful CI run on `main`, builds the Docker image, and pushes it to DockerHub with the `latest` tag and a commit-SHA tag. Actual deployment to a target environment is a manual step.

---

## Configuration

Key parameters are centralised in `config.yaml`:

```yaml
hf_model:
  name: cardiffnlp/twitter-roberta-base-sentiment-latest
  num_labels: 3

hf_hub_model_id: Emanueleb/twitter-roberta-base-sentiment-latest-finetuned

quality_thresholds:
  f1_min: 0.7
  accuracy_min: 0.7

kaggle_dataset:
  name: mdismielhossenabir/sentiment-analysis
  file_path: sentiment_analysis.csv
```

Environment variables (see `.env.example`):

```
HF_TOKEN=<huggingface_token>
MODEL_OUTPUT_DIR=<local_model_output_path>
MLFLOW_EXPERIMENT_NAME=<experiment_name>
HOST=127.0.0.1
PORT=3000
```

---

## Getting Started

```bash
# Clone the repository
git clone <repo-url>
cd sentiment-mlops

# Copy and fill in environment variables
cp .env.example .env

# Start all services
cd docker-env
docker compose up --build
```

Services:
- FastAPI: `http://localhost:8000`
- MLflow UI: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (default credentials: `admin` / `admin`)

---

## Future Improvements

- **Custom dataset loading**: allow users to provide their own dataset at training time to enable retraining on domain-specific data without modifying the codebase.
- **Code quality**: introduce design patterns (e.g. Repository, Strategy, Factory) to make the codebase more maintainable and scalable as the number of supported models and data sources grows.
- **Model selection for inference**: expose a mechanism to choose which model version to use for prediction — the base model, the latest fine-tuned model, or a specific version from the fine-tuning history stored on Hugging Face Hub.
