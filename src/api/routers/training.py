from fastapi import APIRouter, HTTPException, Request
from api.schemas.training import TrainingResponse
from api.services.training import train_and_save_model
from api.utils.metrics import model_f1_macro, model_accuracy, model_eval_loss
from predicting.make_prediction import create_sentiment_pipeline

router = APIRouter(
    prefix="/train",
    tags=["Training"]
)


@router.post("")
async def train(request: Request) -> TrainingResponse:
    try:
        app_config = request.app.state.config
        metrics = train_and_save_model(app_config)
        model_f1_macro.set(metrics["f1_macro"])
        model_accuracy.set(metrics["accuracy"])
        model_eval_loss.set(metrics["loss"])
        hub_model_id = app_config["hf_hub_model_id"]
        app_config["sentiment_pipeline"] = create_sentiment_pipeline(
            hub_model_id
        )
        app_config["prediction_model_name"] = hub_model_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return TrainingResponse(status="Training completed")
