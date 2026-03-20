from fastapi import APIRouter, Request
from api.schemas.training import TrainingResponse
from api.services.training import train_and_save_model

router = APIRouter(
    prefix="/train", 
    tags=["Training"]
)

@router.post("")
async def train(request: Request) -> TrainingResponse:
    app_config = request.app.state.config

    train_and_save_model(app_config)

    return TrainingResponse(status="Training completed")
