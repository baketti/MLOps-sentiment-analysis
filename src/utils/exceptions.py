class InvalidDatasetStructureError(Exception):
    """
        Exception raised when the dataset structure is invalid
        or does not contain the required columns.
    """
    pass


class ConfigLoadError(Exception):
    """
        Exception raised when there is an error loading the configuration.
    """
    pass


class LoadingDatasetError(Exception):
    """
        Exception raised when there is an error loading the dataset.
    """
    pass


class FineTuningError(Exception):
    """
        Exception raised when there is an error during fine-tuning.
    """
    pass


class PushingToHubError(Exception):
    """
        Exception raised when there is an error pushing the model
        to the Hugging Face Hub.
    """
    pass


class ModelLoadingError(Exception):
    """
        Exception raised when there is an error loading a model
        or tokenizer.
    """
    pass


class PredictionError(Exception):
    """
        Exception raised when there is an error during prediction.
    """
    pass


class EvaluationError(Exception):
    """
        Exception raised when there is an error during model evaluation.
    """
    pass
