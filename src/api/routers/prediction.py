from fastapi import APIRouter, HTTPException
from predicting.make_prediction import make_prediction
from utils.config import load_config
from api.schemas.prediction import PredictRequest, PredictResponse
from api.utils.utilities import resolve_model

router = APIRouter(
    prefix="/predict", 
    tags=["Prediction"]
)

config = load_config()
BASE_MODEL = config["hf_model"]["name"]
FINETUNED_MODEL = config["hf_hub_model_id"]

@router.post("")
def predict(request: PredictRequest) -> PredictResponse:
    try:
        model_name = resolve_model()
        result = make_prediction(request.text, model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(
        model_used=model_name,
        label=result["label"],
        score=result["score"],
    )
