from pydantic import BaseModel, Field
class PredictRequestBody(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=512,
        description="The input text for sentiment analysis. Must be maximum 512 characters.",
        example="I love this product! It works great and exceeded my expectations.",
    )

class PredictResponseBody(BaseModel):
    model_used: str = Field(
        description="The name of the model used for prediction, either the base model or the fine-tuned model.",
        example="distilbert/base-uncased-finetuned-sst-2-english",
    )
    label: str = Field(
        description="The predicted sentiment label, either 'positive', 'neutral' or 'negative'.",
        example="positive",
    )
    score: float = Field(
        description="The confidence score of the prediction, ranging from 0 to 1.",
        example=0.95,
    )   
