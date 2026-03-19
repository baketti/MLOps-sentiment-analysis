from transformers import pipeline, TextClassificationPipeline
from utils.exceptions import PredictionError

def create_sentiment_pipeline(model_name: str) -> TextClassificationPipeline:
    """
        Creates a Hugging Face pipeline for text classification using the specified model and tokenizer.
        Returns:
            TextClassificationPipeline: A Hugging Face pipeline for text classification.
    """
    try:
        return pipeline(
            task="text-classification",
            model=model_name,
            tokenizer=model_name,
            truncation=True,
            max_length=512,
        )
    except Exception as e:
        raise PredictionError(f"Error loading model '{model_name}': {e}")


def predict(text: str, pipeline: TextClassificationPipeline) -> dict:
    """
        Predicts the sentiment of the given text using the provided pipeline.
        Params:
            text (str): The input text to analyze.
            pipeline (TextClassificationPipeline): The Hugging Face pipeline for text classification.
        Returns:
            dict: A dictionary containing the predicted label and its corresponding score.
    """
    try:
        scores = pipeline(text)
        best = max(scores, key=lambda x: x["score"])
        return best
    except Exception as e:
        raise PredictionError(f"Error during prediction: {e}")


def make_prediction(text: str, model_name: str) -> dict:
    """
        Makes a sentiment prediction for the given text.
        Params:
            text (str): The input text to analyze.
            model_name (str): The name of the model to use for prediction.
        Returns:
            dict: A dictionary containing the predicted label and its corresponding score.
    """
    sentiment_pipeline = create_sentiment_pipeline(model_name)
    prediction = predict(text, sentiment_pipeline)
    return prediction
