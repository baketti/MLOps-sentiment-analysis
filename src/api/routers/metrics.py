from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from api.schemas.training import TrainingMetrics
from api.utils.metrics import model_f1_macro, model_accuracy, model_eval_loss

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


@router.post("/training")
def update_training_metrics(metrics: TrainingMetrics):
    model_accuracy.set(metrics.accuracy)
    model_f1_macro.set(metrics.f1_macro)
    model_eval_loss.set(metrics.loss)
    return {"status": "ok"}
