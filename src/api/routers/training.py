from fastapi import APIRouter, Request
from training.train_model import train_and_save_model
from api.schemas.training import TrainingResponse

router = APIRouter(
    prefix="/train", 
    tags=["Training"]
)

@router.post("")
async def train(request: Request) -> TrainingResponse:
    app_config = request.app.state.config

    model = app_config.get("model_object")
    tokenizer = app_config.get("tokenizer_object")
    dataset_name = app_config.get("kaggle_dataset").get("name")
    file_path = app_config.get("kaggle_dataset").get("file_path")
    label2id = app_config.get("hf_model").get("label2id")
    model_output_dir = app_config.get("model_output_dir")
    print(f"model_output_dir in training endpoint: {model_output_dir}")
    hub_model_id = app_config.get("hf_hub_model_id")
    quality_thresholds = app_config.get("quality_thresholds")

    train_and_save_model(model, tokenizer, dataset_name, file_path, label2id, model_output_dir, hub_model_id, quality_thresholds)

    return TrainingResponse(status="Training completed")
