from utils.config import load_config
from huggingface_hub import model_info

config = load_config()
BASE_MODEL = config["hf_model"]["name"]
FINETUNED_MODEL = config["hf_hub_model_id"]

def resolve_model() -> str:
    try:
        model_info(FINETUNED_MODEL)
        return FINETUNED_MODEL
    except Exception:
        return BASE_MODEL
    