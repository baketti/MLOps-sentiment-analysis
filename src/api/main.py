from utils.config import load_config
from contextlib import asynccontextmanager
from fastapi import FastAPI
from api.routers import training, prediction
from utils.config import load_config

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
        Application lifespan: load configuration once and attach it to app.state.
    """
    app.state.config = load_config()
    yield
    app.state.config.clear()

app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)

app.include_router(training.router)
app.include_router(prediction.router)

@app.get("/config")
async def get_config():
    """
        Return the currently loaded configuration (or a helpful message).

        The config is attached to `app.state` during application startup.
    """
    cfg = getattr(app.state, "config", None)
    return cfg or {"message": "Configuration not loaded yet."}