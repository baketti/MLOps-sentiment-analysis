from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from api.schemas.training import TrainingMetrics
from pydantic import BaseModel
from api.utils.metrics import (
    model_f1_macro, model_accuracy, model_eval_loss,
    model_precision_per_class, model_recall_per_class, model_f1_per_class,
    drift_score,
)

router = APIRouter(
    prefix="/metrics",
    tags=["Metrics"],
)


@router.get("")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


class DriftUpdate(BaseModel):
    score: float


@router.post("/drift")
def update_drift_score(payload: DriftUpdate):
    drift_score.set(payload.score)
    return {"status": "ok"}


@router.post("/training")
def update_training_metrics(metrics: TrainingMetrics):
    model_accuracy.set(metrics.accuracy)
    model_f1_macro.set(metrics.f1_macro)
    model_eval_loss.set(metrics.loss)
    for label in ("negative", "neutral", "positive"):
        model_precision_per_class.labels(label=label).set(
            getattr(metrics, f"precision_{label}")
        )
        model_recall_per_class.labels(label=label).set(
            getattr(metrics, f"recall_{label}")
        )
        model_f1_per_class.labels(label=label).set(
            getattr(metrics, f"f1_{label}")
        )
    return {"status": "ok"}
