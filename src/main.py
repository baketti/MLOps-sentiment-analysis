from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from dotenv import load_dotenv
from utils.config import load_config
import os
from utils.config import load_config
from utils.exceptions import ConfigLoadError, ModelLoadingError
from api.main import app

load_dotenv()

try:
    config = load_config()
except ConfigLoadError as e:
    print(f"Configuration error: {e}")
    raise SystemExit(1)

MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR")
HF_MODEL_NAME = config.get("hf_model", {}).get("name", "cardiffnlp/twitter-roberta-base-sentiment-latest")

NUM_LABELS = config.get("hf_model", {}).get("num_labels", 3)
LABEL2ID   = config.get("hf_model", {}).get("label2id", {"negative": 0, "neutral": 1, "positive": 2})
ID2LABEL   = {v: k for k, v in LABEL2ID.items()}

try:
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_NAME,
        num_labels=NUM_LABELS,
        ignore_mismatched_sizes=True,
    )
except Exception as e:
    raise ModelLoadingError(f"Error loading model '{HF_MODEL_NAME}': {e}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app=app, host="127.0.0.1", port=3000)