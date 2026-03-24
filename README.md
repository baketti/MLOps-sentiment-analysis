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
| `POST` | `/metrics/training` | Aggiorna i gauge Prometheus con le metriche dell'ultimo training (usato dal DAG Airflow) |

### Flusso di training (`POST /train`)

1. Download del dataset da Kaggle tramite `kagglehub`
2. Suddivisione in train/test (80/20, stratificata)
3. Tokenizzazione con `AutoTokenizer` (`max_length=128`, padding dinamico)
4. Fine-tuning del modello con HuggingFace `Trainer`; le metriche vengono riportate automaticamente a MLflow
5. Valutazione del modello fine-tuned rispetto alle soglie di qualità definite in `config.yaml`:
   - `f1_min: 0.85`
   - `accuracy_min: 0.85`
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
├── dags/
│   └── sentiment_retraining_dag.py    # DAG Airflow: download → fine-tune → update metrics → quality gate → push to hub
├── docker-env/
│   ├── Dockerfile                     # python:3.12-slim, PyTorch CPU-only
│   ├── Dockerfile.airflow             # apache/airflow con dipendenze ML pre-installate
│   └── docker-compose.yml             # FastAPI (8000) + MLflow (5000) + Prometheus (9090) + Grafana (3000) + Airflow (8080)
├── monitoring/
│   ├── prometheus.yml                 # Configurazione scrape Prometheus
│   └── grafana/
│       ├── provisioning/
│       │   ├── dashboards/            # Provisioning automatico dashboard
│       │   └── datasources/           # Provisioning automatico datasource Prometheus
│       └── dashboards/                # JSON delle dashboard Grafana
├── .github/workflows/
│   ├── ci.yml                         # Lint + test unitari + test integrazione ad ogni push
│   └── cd.yml                         # Continuous Delivery: build e push immagine Docker su DockerHub al merge su main
├── config.yaml                        # Nome modello, mapping etichette, soglie qualità, config dataset
├── requirements.txt                   # Dipendenze Python
├── pyproject.toml                     # Configurazione build del pacchetto (setuptools)
└── .env.example                       # Template variabili d'ambiente
```

---

## Test

I test coprono selettivamente le componenti più critiche e testabili del progetto: caricamento e preprocessing del dataset, logica di valutazione e quality gate, endpoint API tramite FastAPI TestClient. La copertura è parziale per la natura del progetto: le parti più rilevanti (fine-tuning, inferenza con modelli HuggingFace, integrazione con servizi esterni come MLflow e Kaggle) richiedono risorse computazionali o dipendenze esterne che rendono il testing automatizzato in CI impraticabile senza mock estensivi.

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
  f1_min: 0.85
  accuracy_min: 0.85

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
- Airflow: `http://localhost:8080` (credenziali di default: `admin` / `admin`)

---

## Monitoraggio e Metriche

### Flusso dei dati

```
FastAPI :8000/metrics          (espone metriche in formato Prometheus)
    ↓  scrape ogni 15s
Prometheus :9090               (raccoglie e archivia time-series)
    ↓  query PromQL
Grafana :3000                  (visualizza i dati)
```

FastAPI espone due tipi di metriche:
- **Counter/Histogram** (`sentiment_predictions_total`, `sentiment_inference_latency_seconds`): aggiornati ad ogni predizione
- **Gauge** (`model_eval_accuracy`, `model_eval_f1_macro`, `model_eval_loss`): aggiornati al termine di ogni training tramite `POST /metrics/training` (chiamato dal DAG Airflow) o `POST /train`

### Datasource Grafana (`$datasource`)

Le dashboard usano la variabile `$datasource`. Grafana la risolve automaticamente cercando il primo datasource Prometheus disponibile.

### Persistenza

| Cosa | Dove vive | `restart` | `down` | `down -v` |
|---|---|---|---|---|
| Dati time-series Prometheus | volume `prometheus_data` | ✅ | ✅ | ❌ |
| Dashboard Grafana (JSON) | bind mount `monitoring/grafana/dashboards/` | ✅ | ✅ | ✅ |
| Datasource Grafana | bind mount `monitoring/grafana/provisioning/` | ✅ | ✅ | ✅ |
| Gauge/Counter FastAPI | memoria processo | ❌ | ❌ | ❌ |

- **`restart`**: volumi intatti, dati Prometheus preservati. I gauge FastAPI si azzerano — su Grafana si vedrà un calo brusco a zero.
- **`down`** (senza `-v`): equivalente a restart, volumi intatti.
- **`down -v`**: distrugge tutti i volumi incluso `prometheus_data`. Tutti i dati storici Prometheus vengono persi.

### Comportamento dopo un riavvio

- **Counter predizioni** (`sentiment_predictions_total`): si azzera. La metrica sparisce da Prometheus finché non viene effettuata almeno una predizione — Grafana mostra "No data" finché Prometheus non raccoglie il primo valore.
- **Gauge di training**: si azzerano a `0`. Grafana mostra `0` fino al prossimo training.

---

## Miglioramenti Futuri

- **Persistenza metriche di training**: al riavvio di FastAPI i gauge Prometheus si azzerano. Possibili soluzioni migliori includono rileggere le metriche dell'ultimo run MLflow al boot per ripopolare i gauge, o persistere i valori su un database esterno (es. PostgreSQL) da cui FastAPI li recupera all'avvio, oppure le metriche potrebbero essere visualizzate direttamente dal DB senza passare per Prometheus.
- **Refactoring pipeline Airflow → FastAPI**: attualmente il container Airflow installa autonomamente le stesse dipendenze ML (PyTorch, Transformers, ecc.) già presenti nel container FastAPI, rendendo il build complessivamente più oneroso in termini di tempo e risorse. Possibili soluzioni migliori includono esporre endpoint dedicati su FastAPI per ogni step della pipeline (`/pipeline/download`, `/pipeline/fine-tune`, ecc.) in modo che i task Airflow eseguano semplici chiamate HTTP, eliminando la necessità di installare dipendenze ML anche nel container Airflow.
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
| `POST` | `/metrics/training` | Updates Prometheus gauges with the latest training metrics (called by the Airflow DAG) |

### Training flow (`POST /train`)

1. Download the dataset from Kaggle via `kagglehub`
2. Split into train/test sets (80/20, stratified)
3. Tokenize with `AutoTokenizer` (`max_length=128`, dynamic padding)
4. Fine-tune the model using HuggingFace `Trainer`; metrics are reported to MLflow automatically
5. Evaluate the fine-tuned model against the quality thresholds defined in `config.yaml`:
   - `f1_min: 0.85`
   - `accuracy_min: 0.85`
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
├── dags/
│   └── sentiment_retraining_dag.py    # Airflow DAG: download → fine-tune → update metrics → quality gate → push to hub
├── docker-env/
│   ├── Dockerfile                     # python:3.12-slim, CPU-only PyTorch
│   ├── Dockerfile.airflow             # apache/airflow with pre-installed ML dependencies
│   └── docker-compose.yml             # FastAPI (8000) + MLflow (5000) + Prometheus (9090) + Grafana (3000) + Airflow (8080)
├── monitoring/
│   ├── prometheus.yml                 # Prometheus scrape configuration
│   └── grafana/
│       ├── provisioning/
│       │   ├── dashboards/            # Automatic dashboard provisioning
│       │   └── datasources/           # Automatic Prometheus datasource provisioning
│       └── dashboards/                # Grafana dashboard JSON files
├── .github/workflows/
│   ├── ci.yml                         # Lint + unit + integration tests on every push
│   └── cd.yml                         # Continuous Delivery: builds and pushes Docker image to DockerHub on main
├── config.yaml                        # Model name, label mapping, quality thresholds, dataset config
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Package build config (setuptools)
└── .env.example                       # Environment variable template
```

---

## Testing

Tests cover only a subset of the most critical project components. Coverage is intentionally partial given the educational nature of the project and serves primarily to demonstrate the continuous integration pipeline in action.

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
  f1_min: 0.85
  accuracy_min: 0.85

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
- Airflow: `http://localhost:8080` (default credentials: `admin` / `admin`)

---

## Monitoring and Metrics

### Data flow

```
FastAPI :8000/metrics          (exposes metrics in Prometheus format)
    ↓  scrape every 15s
Prometheus :9090               (collects and stores time-series)
    ↓  PromQL query
Grafana :3000                  (visualises the data)
```

FastAPI exposes two types of metrics:
- **Counter/Histogram** (`sentiment_predictions_total`, `sentiment_inference_latency_seconds`): updated on every prediction
- **Gauges** (`model_eval_accuracy`, `model_eval_f1_macro`, `model_eval_loss`): updated at the end of each training run via `POST /metrics/training` (called by the Airflow DAG) or `POST /train`

### Grafana datasource (`$datasource`)

Dashboards use the `$datasource` template variable instead of a hardcoded UID. Grafana resolves it automatically by finding the first available Prometheus datasource — regardless of the internally generated UID. This makes dashboards portable across any installation without modifying the JSON files.

### Persistence

| What | Where it lives | `restart` | `down` | `down -v` |
|---|---|---|---|---|
| Prometheus time-series data | `prometheus_data` volume | ✅ | ✅ | ❌ |
| Grafana dashboards (JSON) | bind mount `monitoring/grafana/dashboards/` | ✅ | ✅ | ✅ |
| Grafana datasource | bind mount `monitoring/grafana/provisioning/` | ✅ | ✅ | ✅ |
| FastAPI Gauges/Counters | process memory | ❌ | ❌ | ❌ |

- **`restart`**: volumes intact, Prometheus data preserved. FastAPI gauges reset — Grafana shows a sharp drop to zero.
- **`down`** (without `-v`): equivalent to restart, volumes intact.
- **`down -v`**: destroys all volumes including `prometheus_data`. All Prometheus historical data is lost.

### Behaviour after a restart

- **Prediction counter** (`sentiment_predictions_total`): resets to zero. The metric disappears from Prometheus until at least one prediction is made — Grafana shows "No data" until Prometheus collects the first value.
- **Training gauges**: reset to `0`. Grafana shows `0` until the next training run.

---

## Future Improvements

- **Training metrics persistence**: on FastAPI restart, Prometheus gauges reset to zero. Possible better solutions include reading the latest MLflow run metrics at boot to repopulate the gauges, or persisting values in an external database (e.g. PostgreSQL) from which FastAPI recovers them on startup, or metrics could be visualised directly from the database without going through Prometheus.
- **Airflow → FastAPI pipeline refactoring**: the Airflow container currently installs the same ML dependencies (PyTorch, Transformers, etc.) already present in the FastAPI container, making the overall build more expensive in terms of time and resources. Possible better solutions include exposing dedicated FastAPI endpoints for each pipeline step (`/pipeline/download`, `/pipeline/fine-tune`, etc.) so that Airflow tasks make simple HTTP calls, eliminating the need to install ML dependencies in the Airflow container.
- **Custom dataset loading**: allow users to provide their own dataset at training time to enable retraining on domain-specific data without modifying the codebase.
- **Code quality**: introduce design patterns (e.g. Repository, Strategy, Factory) to make the codebase more maintainable and scalable as the number of supported models and data sources grows.
- **Model selection for inference**: expose a mechanism to choose which model version to use for prediction — the base model, the latest fine-tuned model, or a specific version from the fine-tuning history stored on Hugging Face Hub.
