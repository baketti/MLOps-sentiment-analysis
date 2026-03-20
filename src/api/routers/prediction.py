from fastapi import APIRouter, HTTPException, Request
from predicting.make_prediction import make_prediction
from utils.config import load_config
from api.schemas.prediction import PredictRequestBody, PredictResponseBody
from api.utils.utilities import resolve_model

router = APIRouter(
    prefix="/predict", 
    tags=["Prediction"]
)

@router.post("")
async def predict(payload: PredictRequestBody, request: Request) -> PredictResponseBody:
    try:
        app_config = request.app.state.config

        base_model_name = app_config["hf_model"]["name"]
        finetuned_model_name = app_config["hf_hub_model_id"]

        model_name = resolve_model(base_model_name, finetuned_model_name)
        result = make_prediction(payload.text, model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponseBody(
        model_used=model_name,
        label=result["label"],
        score=result["score"],
    )
