from predicting.make_prediction import predict


def make_prediction(app_config, text: str) -> dict:
    """
        Makes a sentiment prediction for the given text using the
        pipeline cached at startup (or updated after training).
        Params:
            app_config (dict): The application configuration dictionary
                containing all necessary parameters for prediction.
            text (str): The input text to analyze.
        Returns:
            prediction (dict): A dictionary containing the predicted label and
                its corresponding score.
            model_name (str): The name of the model used for prediction.
        Raises:
            PredictionError: If there is an error during prediction.
    """
    sentiment_pipeline = app_config["sentiment_pipeline"]
    model_name = app_config["prediction_model_name"]
    prediction = predict(text, sentiment_pipeline)
    return prediction, model_name
