from fastapi import APIRouter
from training.train_model import model, tokenizer, MODEL_OUTPUT_DIR, train_and_save_model

router = APIRouter(
    prefix="/train", 
    tags=["Training"]
)

@router.post("")
def train():
    train_and_save_model(model, tokenizer, MODEL_OUTPUT_DIR)
    return {"status": "Training completed"}
