from fastapi import APIRouter, HTTPException, Request
from api.schemas.training import TrainingResponse
from api.services.training import train_and_save_model

router = APIRouter(
    prefix="/train",
    tags=["Training"]
)

@router.post("")
async def train(request: Request) -> TrainingResponse:
    try:
        app_config = request.app.state.config
        train_and_save_model(app_config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return TrainingResponse(status="Training completed")
