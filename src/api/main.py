import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline as hf_pipeline,
)
from utils.config import load_config
from utils.exceptions import ConfigLoadError, ModelLoadingError
from api.routers import training, prediction, metrics
from api.utils.utilities import resolve_model
from predicting.make_prediction import create_sentiment_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
        Application lifespan: load configuration once
        and attach it to app.state.
    """
    try:
        load_dotenv()
        MODEL_OUTPUT_DIR = os.getenv("MODEL_OUTPUT_DIR")
        print(f"MODEL_OUTPUT_DIR from dotenv: {MODEL_OUTPUT_DIR}")
        config = load_config()

        app.state.config = config
        app.state.config['model_output_dir'] = MODEL_OUTPUT_DIR
        print(
            f"MODEL_OUTPUT_DIR in app.state.config: "
            f"{app.state.config['model_output_dir']}"
        )
        hf_model_config = config.get("hf_model")
        HF_MODEL_NAME = hf_model_config.get("name")
        NUM_LABELS = hf_model_config.get("num_labels")

        try:
            tokenizer: AutoTokenizer = (
                AutoTokenizer.from_pretrained(HF_MODEL_NAME)
            )
            model: AutoModelForSequenceClassification = (
                AutoModelForSequenceClassification.from_pretrained(
                    HF_MODEL_NAME,
                    num_labels=NUM_LABELS,
                    ignore_mismatched_sizes=True,
                )
            )

            app.state.config["tokenizer_object"] = tokenizer
            app.state.config["model_object"] = model

            finetuned_model_name = config.get("hf_hub_model_id")
            prediction_model_name = resolve_model(
                HF_MODEL_NAME, finetuned_model_name
            )
            if prediction_model_name == HF_MODEL_NAME:
                sentiment_pipeline = hf_pipeline(
                    task="text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    truncation=True,
                    max_length=512,
                )
            else:
                sentiment_pipeline = create_sentiment_pipeline(
                    prediction_model_name
                )
            app.state.config["sentiment_pipeline"] = sentiment_pipeline
            app.state.config["prediction_model_name"] = prediction_model_name

        except Exception as e:
            raise ModelLoadingError(
                f"Error loading model '{HF_MODEL_NAME}': {e}"
            )
    except (ConfigLoadError, ModelLoadingError) as e:
        print(f"Startup error: {e}")
        raise
    yield
    app.state.config.clear()


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)

app.include_router(training.router)
app.include_router(prediction.router)
app.include_router(metrics.router)
