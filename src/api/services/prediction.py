from predicting.make_prediction import create_sentiment_pipeline, predict
from api.utils.utilities import resolve_model
from utils.exceptions import PredictionError


def make_prediction(app_config, text: str) -> dict:
    """
        Makes a sentiment prediction for the given text using the
        appropriate model based on the application configuration.
        Params:
            app_config (dict): The application configuration dictionary
                containing all necessary parameters for prediction.
            text (str): The input text to analyze.
        Returns:
            dict: A dictionary containing the predicted label and its
                corresponding score.
    """
    try:
        base_model_name = app_config["hf_model"]["name"]
        finetuned_model_name = app_config["hf_hub_model_id"]

        model_name = resolve_model(base_model_name, finetuned_model_name)
        sentiment_pipeline = create_sentiment_pipeline(model_name)
        prediction = predict(text, sentiment_pipeline)

        return prediction
    except PredictionError as e:
        raise e
    except Exception as e:
        raise e
