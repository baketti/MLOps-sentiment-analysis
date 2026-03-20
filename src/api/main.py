from utils.config import load_config
from utils.exceptions import ConfigLoadError
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routers import training, prediction
from utils.config import load_config
import os
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from utils.exceptions import ConfigLoadError, ModelLoadingError
from dotenv import load_dotenv

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
        Application lifespan: load configuration once and attach it to app.state.
    """
    try:
        load_dotenv()
        MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR")
        print(f"MODEL_OUTPUT_DIR from dotenv: {MODEL_OUTPUT_DIR}")
        config = load_config()

        app.state.config = config
        app.state.config['model_output_dir'] = MODEL_OUTPUT_DIR
        print(f"MODEL_OUTPUT_DIR in app.state.config: {app.state.config['model_output_dir']}")
        hf_model_config = config.get("hf_model")
        HF_MODEL_NAME = hf_model_config.get("name")
        NUM_LABELS = hf_model_config.get("num_labels")

        try:
            tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
            model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(
                HF_MODEL_NAME,
                num_labels=NUM_LABELS,
                ignore_mismatched_sizes=True,
            )

            app.state.config["tokenizer_object"] = tokenizer
            app.state.config["model_object"] = model

        except Exception as e:
            raise ModelLoadingError(f"Error loading model '{HF_MODEL_NAME}': {e}")
    except (ConfigLoadError, ModelLoadingError) as e:
        print(f"Startup error: {e}")
        raise
    yield
    app.state.config.clear()

app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)

app.include_router(training.router)
app.include_router(prediction.router)