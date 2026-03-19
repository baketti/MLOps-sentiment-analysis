from utils.config import load_config
from huggingface_hub import model_info

config = load_config()
BASE_MODEL = config["hf_model"]["name"]
FINETUNED_MODEL = config["hf_hub_model_id"]

def resolve_model() -> str:
    try:
        model_info(FINETUNED_MODEL)
        return FINETUNED_MODEL
    except Exception as e:
        print(f"Fine-tuned model '{FINETUNED_MODEL}' not available: {e}. Falling back to base model.")
        return BASE_MODEL
    