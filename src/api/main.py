from fastapi import FastAPI
from api.routers import training, prediction

app = FastAPI(title="Sentiment Analysis API")

app.include_router(training.router)
app.include_router(prediction.router)
