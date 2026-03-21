from pydantic import BaseModel, Field


class PredictRequestBody(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=512,
        description=(
            "The input text for sentiment analysis."
            " Must be maximum 512 characters."
        ),
    )


class PredictResponseBody(BaseModel):
    model_used: str = Field(
        description=(
            "The name of the model used for prediction,"
            " either the base model or the fine-tuned model."
        ),
    )
    label: str = Field(
        description=(
            "The predicted sentiment label,"
            " either 'positive', 'neutral' or 'negative'."
        ),
    )
    score: float = Field(
        description=(
            "The confidence score of the prediction, ranging from 0 to 1."
        ),
    )
