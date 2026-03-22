from fastapi import APIRouter, HTTPException, Request
from api.schemas.prediction import PredictRequestBody, PredictResponseBody
from api.services.prediction import make_prediction

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)


@router.post("")
async def predict(
    payload: PredictRequestBody, request: Request
) -> PredictResponseBody:
    try:
        app_config = request.app.state.config
        result, model_name = make_prediction(app_config, payload.text)

        return PredictResponseBody(
            model_used=model_name,
            label=result["label"],
            score=result["score"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
