from huggingface_hub import model_info
from huggingface_hub.errors import RepositoryNotFoundError


def resolve_model(base_model_name: str, finetuned_model_name: str) -> str:
    """
        Resolves which model to use for prediction. It first checks if
        the fine-tuned model is available on Hugging Face Hub.
        If it is, it returns the fine-tuned model name.
        If not, it falls back to the base model name.
        Params:
            base_model_name (str): The name of the base model.
            finetuned_model_name (str): The name of the fine-tuned model.
        Returns:
            str: The name of the model to use for prediction.
    """
    try:
        model_info(finetuned_model_name)
        return finetuned_model_name
    except RepositoryNotFoundError as e:
        print(
            f"Fine-tuned model '{finetuned_model_name}' not available: {e}."
            f" Falling back to base model."
        )
        return base_model_name
